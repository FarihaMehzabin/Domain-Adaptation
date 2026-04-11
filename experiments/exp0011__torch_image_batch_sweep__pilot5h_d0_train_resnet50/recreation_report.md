# Torch Image Batch Sweep

- Experiment: `exp0011__torch_image_batch_sweep__pilot5h_d0_train_resnet50`
- Domain/split: `d0_nih/train`
- Encoder: `torchvision:resnet50`
- Weights: `DEFAULT`
- Candidates: `[256, 512, 768, 1024, 1280, 1536, 1792, 2048]`
- Highest successful batch size: `1536`

## Results
- batch_size=256: success, elapsed_seconds=12.685123037546873, seconds_per_image=0.04955126186541747, peak_memory_mb=2793.72802734375
- batch_size=512: success, elapsed_seconds=21.226034611463547, seconds_per_image=0.04145709885051474, peak_memory_mb=5487.72802734375
- batch_size=768: success, elapsed_seconds=30.574507616460323, seconds_per_image=0.03981055679226605, peak_memory_mb=8183.72802734375
- batch_size=1024: success, elapsed_seconds=48.026415321975946, seconds_per_image=0.046900796212867135, peak_memory_mb=10877.72802734375
- batch_size=1280: success, elapsed_seconds=57.19159311056137, seconds_per_image=0.04468093211762607, peak_memory_mb=13573.72802734375
- batch_size=1536: success, elapsed_seconds=65.56414415687323, seconds_per_image=0.04268498968546434, peak_memory_mb=16267.72802734375
- batch_size=1792: failure, elapsed_seconds=80.43017750605941, seconds_per_image=0.04488291155472065, peak_memory_mb=13475.72802734375, error_type=OutOfMemoryError
