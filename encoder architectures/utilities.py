
def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}

def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)

def build_model(config: ModelConfig, checkpoint: str | Path | None = None, device: str | torch.device | None = None) -> NeighborAttentionValuePolicyNet:
    model = NeighborAttentionValuePolicyNet(config)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("model", payload)
        model.load_state_dict(_strip_compile_prefix(state), strict=False)
    return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
