# MLXImageGeneration

Text-to-image models, generated one bounded step at a time.

The library supports [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1). Its
distilled Turbo variant shares the architecture and ships its own sampling grid, which is
read from the checkpoint. A checkpoint's weights carry their own license, separate from this
code. Check it before you ship one.

## Generating an image

Load a checkpoint from a local directory, then drive an `ImageIterator`:

```swift
import MLXHuggingFace
import MLXImageGeneration

let model = try await QwenImageModel.load(
    from: checkpointDirectory, tokenizerLoader: #huggingFaceTokenizerLoader())

var iterator = try ImageIterator(
    model: model,
    prompt: "a red fox sitting in fresh snow, photograph",
    parameters: .init(width: 1024, height: 1024, stepCount: 40, seed: 7))

while let event = iterator.next() {
    switch event {
    case .promptEncoded:
        break
    case .denoised(let step, let total):
        print("step \(step) of \(total)")
    case .image(let pixels):
        // RGBA bytes, [height, width, 4]
        let bytes = pixels.asArray(UInt8.self)
    }
}
```

Each `next()` runs one stage to completion on the GPU and then returns: the text encoder,
then one denoising step per call, then the image decoder. Nothing is left running between
calls. A caller that also runs a language model or an embedding model decides what the GPU
does between two steps.

## Parameters

- `width`, `height`: multiples of 32.
- `stepCount`: 40 for the base model. A checkpoint that ships its own sampling grid, as Turbo
  does with eight levels, uses the grid and ignores this.
- `guidanceScale`: above 1, each step also runs `negativePrompt` and moves away from it,
  which doubles the step's work.
- `seed`: the same seed, size and prompt give the same image.

## Checkpoints

`QwenImageModel.load` reads a directory with `text_encoder`, `transformer`, `vae`,
`scheduler` and `processor` folders. Both layouts load:

- the original `Qwen/Qwen-Image-2.1` at full precision, about 33 GB
- an MLX conversion, such as the 4-bit `mlx-community/Qwen-Image-2.1-MLX-4bit`, about 10 GB

All three components stay in memory once loaded.

## Accuracy on Metal

The image decoder runs in float32. Two things in MLX affect its accuracy on the GPU:

- Large 3x3 convolutions take a Winograd path that is off by about 6e-3 per layer. The
  decoder avoids that path by padding one input channel, which costs some speed.
- With `MLX_ENABLE_TF32` at its default of 1, matrix products round more coarsely. Decoding
  random latents then differs from the CPU result by about 1e-4 on average. Set
  `MLX_ENABLE_TF32=0` in the environment for a decode that matches the CPU.

## Not supported

Image editing with reference images, ControlNet, LoRA adapters and tiled decoding.
