When DataLoader requests index 500, __getitem__ looks up self.samples[500] which gives a file path and label number. It then opens that file from disk using Image.open(), converts it to RGB, applies the transforms — resize, ToTensor, normalize — and returns a tensor of shape (3, 224, 224) along with the integer label. The DataLoader collects 32 of these and stacks them into a batch of (32, 3, 224, 224).

"I initially used random_split which splits by index in memory without moving files. The problem is both splits share the same underlying dataset object, so you can't give them different transforms cleanly — changing val's transform also affects train. The disk split solves this by loading two completely separate ImageFolder objects, each with their own transform pipeline. Val gets only resize and normalize, train gets full augmentation.

"Train transforms include augmentation — random flips, rotations, color jitter — to show the model varied versions of each image so it learns to generalise. Val transforms only resize and normalize because val is a measurement tool. If you augment val the accuracy score changes randomly each run making it impossible to know whether the model actually improved between epochs. Val must be deterministic — same input every time — so you can trust the numbers."


ToTensor does three things. First it converts the PIL image object into a PyTorch tensor so the model can do math on it. Second it changes the shape from (H, W, C) to (C, H, W) — channels move from last position to first because PyTorch expects channels first. Third it rescales pixel values from 0–255 to 0.0–1.0 by dividing by 255. The -2 to +2 range comes separately from Normalize which shifts values using ImageNet mean and std."

"A conv layer slides a small filter — typically 3×3 — across the entire image. At each position it does element-wise multiplication between the filter values and the image patch underneath, then sums all 9 results into one number. That number represents how strongly that pattern was present at that location. Doing this for every position produces a feature map. Negative values mean the pattern was absent, so ReLU after conv removes them by setting negatives to zero. In my model I used padding=1 so the spatial size stays the same after conv — only MaxPool shrinks the dimensions.


"Each block has a conv layer that increases the number of channels — 3→32→64→128 — and a MaxPool that halves the spatial size — 224→112→56→28. The reason for 3 blocks is that each block learns increasingly complex features. Block 1 detects simple things like edges and colour patches directly from pixels. Block 2 combines those edges into shapes and spots. Block 3 combines those shapes into disease-specific patterns like lesions or infected leaf areas. One conv layer alone can only detect simple patterns — depth is what allows the model to understand complex visual concepts."


"A CNN processes images by sliding filters over local patches and stacking layers to build from simple to complex features. Each layer only sees a small neighbourhood. A Vision Transformer splits the image into fixed patches, embeds each patch as a vector, then uses self-attention so every patch can directly relate to every other patch from the very first layer. CNN builds global understanding gradually through depth. Transformer gets global context immediately through attention. For my dataset of 9,102 images CNN was the better choice — Vision Transformers need much larger datasets to train effectively from scratch because they have no built-in assumption about spatial locality.
 

**Freeze** when the dataset is small and the pretrained features are already useful. This reduces overfitting and speeds up training.
**Fine-tune** when enough data is available and the target task differs from the original pretrained task. This allows the network to adapt its learned features to the new domain.

Freezing means "use the pretrained knowledge as-is." Fine-tuning means "allow the pretrained knowledge to adapt to my dataset."