def train_stable_avi(initial_model: NeighborAttentionValuePolicyNet, model_config: ModelConfig, config: StableAVIConfig = StableAVIConfig(), *, log_callback: Callable[[int, dict[str, float]], None] | None = None) -> NeighborAttentionValuePolicyNet:
    if model_config.n != config.n: raise ValueError(f"model_config.n={model_config.n} must equal config.n={config.n}")
    set_seed(config.seed); work_dir = Path(config.work_dir); work_dir.mkdir(parents=True, exist_ok=True); device = torch.device(config.device)
    online = initial_model.to(device); target = build_model(model_config, device=device); target.load_state_dict(unwrap_compiled_model(online).state_dict()); target.eval()
    replay = SourceWeightedReplayBuffer(config.replay_capacity, n=config.n, source_weights=config.source_weights, prioritized=True, freshness_half_life=max(1, config.iterations // 2), pin_memory=str(config.device).startswith("cuda"))
    anchor_source_id = replay.anchor_source_id; _seed_replay(replay, target, config)
    optimizer = torch.optim.AdamW(online.parameters(), lr=config.lr, weight_decay=config.weight_decay); scaler = grad_scaler(config.device, config.amp); device_type = "cuda" if str(config.device).startswith("cuda") else "cpu"
    if config.debug_memory and torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(config.device)
    pbar = tqdm(range(1, config.iterations + 1), desc=f"stable-avi-gnn-rl-n{config.n}"); last_walk_steps = (config.seed_walk_length_min, config.seed_walk_length_max); last_gap_steps = (0, 0, 0); model_guided_active = False
    
    for iteration in pbar:
        progress = iteration / max(1, config.iterations)
        if iteration <= config.bellman_warmup_iterations: policy_fraction = 0.3
        else: anneal_progress = min(1.0, (iteration - config.bellman_warmup_iterations) / 500); policy_fraction = min(0.5 + 0.4 * anneal_progress, 0.99)
        gap_cfg = GapAscentConfig(); gap_active = iteration >= gap_cfg.start_iter; walk_batch_size = gap_cfg.batch_size if gap_active else config.random_walk_width
        if iteration == 1 or iteration % 10 == 0:
            _add_anchor_batch(replay, config, progress=progress)
            last_walk_steps = _add_walk_curriculum(replay, config, count=config.random_walk_width, source="walk_curriculum", progress=progress, iteration=iteration, gap_batch_size=walk_batch_size if gap_active else None)
        last_gap_steps = _inject_gap_curriculum(replay, gap_cfg, iteration, config.n, device, dummy_target_base=float(config.anchor_shell_depth + 2))
        
        total_loss, total_value_loss, total_policy_loss = 0.0, 0.0, 0.0; micro = max(1, min(config.train_microbatch_size, config.batch_size))
        
        for _ in range(config.updates_per_iteration):
            # === 1. Сэмплируем базовый батч ===
            base_batch = config.batch_size // 2
            hard_batch = config.batch_size - base_batch
            states, replay_targets, sample_weights, sampled_source_ids = replay.sample(base_batch, device=device)
            source_is_anchor = (sampled_source_ids == anchor_source_id)
            
            # === 2. Добавляем сложные примеры из hardness buffer ===
            if hard_buffer and hard_batch > 0 and random.random() < 0.3:
                k = min(hard_batch, len(hard_buffer))
                indices = random.choices(range(len(hard_buffer)), weights=hard_priorities, k=k)
                hard_states = torch.stack([hard_buffer[i] for i in indices], dim=0).to(device)
                hard_tgts = torch.stack([hard_targets[i] for i in indices], dim=0).to(device) if hard_targets else torch.zeros(k, device=device)
                hard_weights = torch.ones(k, device=device)
                hard_source_ids = torch.full((k,), -1, dtype=torch.long, device=device)
                states = torch.cat([states, hard_states], dim=0)
                replay_targets = torch.cat([replay_targets, hard_tgts], dim=0)
                sample_weights = torch.cat([sample_weights, hard_weights], dim=0)
                sampled_source_ids = torch.cat([sampled_source_ids, hard_source_ids], dim=0)
                source_is_anchor = torch.cat([source_is_anchor, torch.zeros(k, dtype=torch.bool, device=device)], dim=0)
            
            # === 3. Bellman backup ===
            with torch.no_grad():
                td_targets, sampled_actions, sampled_policy = compute_bellman_batch(states, target, bellman=config.bellman, amp=config.amp, policy_fraction=policy_fraction)
                primary_targets = td_targets.clone()
                if iteration <= config.bellman_warmup_iterations: 
                    primary_targets[~source_is_anchor] = replay_targets[~source_is_anchor]
                primary_targets[is_goal(states, config.n).to(device)] = 0.0
            
            # Обновляем success_by_depth ТОЛЬКО на hard states (защита от leakage)
            with torch.no_grad():
                # Вычисляем всё для батча сразу
                gaps = gap_heuristic(states)  # [B]
                depths = torch.clamp((gaps * 1.2).long(), max=int(1.5 * config.n))  # [B]
                td_errors = torch.abs(td_targets - values)  # [B]
                uncertainties = 1.0 - torch.sigmoid(values - config.n/2)  # [B]
                beam_failed = (td_targets > config.n)  # [B]
                
                # Маска hard states
                hard = (td_errors > 0.5) | (uncertainties > 0.5) | beam_failed
                
                # Обновляем frontier только для hard states (минимум CPU-переключений)
                hard_indices = torch.where(hard)[0].cpu().tolist()
                for idx in hard_indices:
                    d = int(depths[idx].item())
                    gap = float(gaps[idx].item())
                    td_err = float(td_errors[idx].item())
                    solved = bool(values[idx].item() < config.n / 4)
                    _update_frontier(d, gap, td_err, solved)
                    
                    # Добавляем в hardness buffer
                    priority = td_err + float(uncertainties[idx].item()) + 2.0 * float(beam_failed[idx].item())
                    hard_buffer.append(states[idx].detach().cpu().clone())
                    hard_priorities.append(priority)
                    hard_targets.append(td_targets[idx].detach().cpu().clone())
                    if len(hard_buffer) > 10_000:
                        idx_min = torch.argmin(torch.tensor(hard_priorities)).item()
                        hard_buffer.pop(idx_min); hard_priorities.pop(idx_min); hard_targets.pop(idx_min)
                    
                    # beam_failure source
                    if beam_failed[idx] and random.random() < 0.1:
                        replay.add(
                            states[idx:idx+1].cpu(), 
                            td_targets[idx:idx+1].cpu(), 
                            confidence=3.0,
                            priorities=torch.tensor([priority]),
                            source="beam_failure",
                            replace_if_lower=False
                        )
            
            # === 5. Градиентный шаг ===
            optimizer.zero_grad(set_to_none=True)
            update_loss, update_value, update_policy = torch.zeros((), device=device), torch.zeros((), device=device), torch.zeros((), device=device)
            denom = sample_weights.mean().clamp_min(1e-6)
            
            for start in range(0, states.shape[0], micro):
                mb_states = states[start : start + micro]
                mb_targets = primary_targets[start : start + micro]
                mb_replay_targets = replay_targets[start : start + micro]
                mb_weights = sample_weights[start : start + micro]
                mb_actions = sampled_actions[start : start + micro]
                mb_policy = sampled_policy[start : start + micro]
                mb_anchor = source_is_anchor[start : start + micro]
                scale = mb_states.shape[0] / states.shape[0]
                
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=scaler.is_enabled()):
                    pred, logits = online.predict_all(mb_states, use_neighbors=True)
                    value_raw = F.smooth_l1_loss(pred, mb_targets, reduction="none")
                    value_loss = (value_raw * mb_weights / denom).mean()
                    
                    if mb_actions.numel() > 0 and mb_policy.numel() > 0:
                        selected_logits = logits.gather(1, (mb_actions - 2).clamp(0, config.n - 2))
                        log_probs = F.log_softmax(selected_logits, dim=1)
                        policy_loss = F.kl_div(log_probs, mb_policy, reduction="batchmean")
                    else:
                        policy_loss = torch.zeros((), device=device)
                    
                    loss = (value_loss + config.policy_loss_weight * policy_loss) * scale
                
                scaler.scale(loss).backward()
                update_loss += loss.detach()
                update_value += value_loss.detach() * scale
                update_policy += policy_loss.detach() * scale
            
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(update_loss.cpu())
            total_value_loss += float(update_value.cpu())
            total_policy_loss += float(update_policy.cpu())
        
        if iteration % config.target_update_interval == 0: ema_update_target(online, target, tau=config.target_ema_tau)
        
        # === 6. Метрики + логирование глубин ===
        metrics = {
            "loss": total_loss/max(1,config.updates_per_iteration), 
            "value_loss": total_value_loss/max(1,config.updates_per_iteration), 
            "policy_loss": total_policy_loss/max(1,config.updates_per_iteration), 
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()), 
            "replay_size": float(len(replay)), 
            "curriculum/random_walk_steps_min": float(last_walk_steps[0]), 
            "curriculum/random_walk_steps_max": float(last_walk_steps[1]), 
            "curriculum/gap_ascent_steps_min": float(last_gap_steps[0]), 
            "curriculum/gap_ascent_steps_max": float(last_gap_steps[1]), 
            "curriculum/gap_ascent_steps_actual": float(last_gap_steps[2]), 
            "curriculum/model_guided_active": float(1.0 if model_guided_active else 0.0), 
            "curriculum/model_guided_steps": float(config.guided_walk_length if model_guided_active else 0), 
            "curriculum/policy_fraction": float(policy_fraction)
        }
        
        # 🔥 Логирование реальных глубин раз в 100 итераций
        if iteration % 100 == 0 and _curriculum_depths:
            recent = _curriculum_depths[-200:]
            dist = Counter(recent)
            top = dist.most_common(5)
            avg_d = sum(recent) / len(recent)
            print(f"\n📊 Depth stats (iter {iteration}): avg={avg_d:.1f}, range=[{min(recent)}, {max(recent)}], top={top}")
            _curriculum_depths.clear()
        
        counts = replay.source_counts()
        for source, count in counts.items(): metrics[f"replay/{source}"] = float(count)
        
        if config.eval_interval > 0 and (iteration == 1 or iteration % config.eval_interval == 0):
            search_cfg = SearchConfig(device=config.device, amp=config.amp)
            # 🔥 Передаём use_frontier=True для честной оценки
            metrics.update(training_diagnostics(online, n=config.n, samples=config.eval_samples, search_config=search_cfg, use_frontier=True))
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                sr=f"{metrics['diag/beam_success_rate']:.2f}",
                plen=f"{metrics['diag/path_len']:.1f}",
                rw=f"{metrics['curriculum/random_walk_steps_min']:.0f}-{metrics['curriculum/random_walk_steps_max']:.0f}",
                ga=f"{metrics['curriculum/gap_ascent_steps_actual']:.0f}",
                mg=f"{metrics['curriculum/model_guided_steps']:.0f}",
                pf=f"{metrics['curriculum/policy_fraction']:.2f}"
            )
        
        if log_callback is not None: log_callback(iteration, metrics)
        if config.checkpoint_interval > 0 and iteration % config.checkpoint_interval == 0: 
            save_checkpoint(work_dir / f"stable_avi_gnn_rl_n{config.n}_iter{iteration}.pt", online, model_config, train_config=config, target_model=target, replay=replay, iteration=iteration)
    
    save_checkpoint(work_dir / f"stable_avi_gnn_rl_n{config.n}_final.pt", online, model_config, train_config=config, target_model=target, replay=replay, iteration=config.iterations)
    return online

train_bellman_rl = train_stable_avi
