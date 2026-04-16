"""Example: extract activations from GPT-2 at specified layers and positions."""

from activation_extractor import ExtractionConfig, make_extractor


def main() -> None:
    # 1. Load model & build extractor
    extractor = make_extractor("gpt2")
    print(f"Model loaded — {extractor.n_layers} layers, device={extractor.model.device}")

    inputs = ["The cat sat on the mat", "Hello world"]

    # 2. All positions, first & last layer
    cfg = ExtractionConfig(layers=[0, extractor.n_layers - 1], positions="all")
    data = extractor.extract(inputs, cfg)
    print(f"\n[all positions]  {data}")
    print(f"  tokens[0]: {data.tokens[0]}")
    print(f"  activations sample [0, layer=0, token=0, :5]: {data.activations[0, 0, 0, :5]}")

    # 3. Last-token only
    cfg_last = ExtractionConfig(layers=[0, 5, 11], positions="last")
    data_last = extractor.extract(inputs, cfg_last)
    print(f"\n[last token]     {data_last}")

    # 4. Mean-pooled
    cfg_mean = ExtractionConfig(layers=[0, 5, 11], positions="mean")
    data_mean = extractor.extract(inputs, cfg_mean)
    print(f"\n[mean pooled]    {data_mean}")

    # 5. Specific token indices
    cfg_idx = ExtractionConfig(layers=[0], positions=[0, 1])
    data_idx = extractor.extract(inputs[0], cfg_idx)
    print(f"\n[indices 0,1]    {data_idx}")

    # 6. All layers (default config)
    data_all = extractor.extract(inputs[0])
    print(f"\n[all layers/all] {data_all}")

    # 7. Access layer module (useful for Jacobian work)
    layer_0 = extractor.get_layer_module(0)
    print(f"\nLayer 0 module type: {type(layer_0).__name__}")


if __name__ == "__main__":
    main()
