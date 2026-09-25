
I suggest splitting tasks related to training and tasks related to experiments, as once trained models can be shared between the team for experiments. I have identified the following models we would need to train to do component-wise analysis and system-wide (unconditional) comparisions:

| Training Iterations | Training Dataset | Vision Model | w. REPA? | Target Repr. (for representation alignment) | Depth | To do |
|--|--|--|--|--|--|--|
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | MAE-L | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINO-B | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | MoCov3-L | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | I-JEPA-H | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | CLIP-L | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | SigLIP-L | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-B | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-g | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 6 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 10 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 12 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 14 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 16 | [ ] |
| 400K | ImageNet 256x256 | DiT-L/2 | No | N/A | N/A | [ ] |
| 400K | ImageNet 256x256 | DiT-XL/2 | No | N/A | N/A | [ ] |
| 7M | ImageNet 256x256 | DiT-XL/2 | No | N/A | N/A | [ ] |
| 400K | ImageNet 256x256 | SiT-B/2 | No | N/A | N/A | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | No | N/A | N/A | [ ] |
| 400K | ImageNet 256x256 | SiT-XL/2 | No | N/A | N/A | [ ] |
| 7M | ImageNet 256x256 | SiT-XL/2 | No | N/A | N/A | [ ] |
| 400K | ImageNet 256x256 | DiT-L/2 | Yes | DINOv2-L? | 8 | [ ] |
| 400K | ImageNet 256x256 | DiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |
| 7M | ImageNet 256x256 | DiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-B/2 | Yes | DINOv2-L? | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L? | 8 | [ ] |
| 700K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L? | 8 | [ ] |
| 150K | ImageNet 256x256 | SiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |
| 400K | ImageNet 256x256 | SiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |
| 1M | ImageNet 256x256 | SiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |

This is probably too much to do in time so we need to decide which ones we will train. In their training script I cannot see an easy option to train DiT models so these are a good candidate to remove from our list.
