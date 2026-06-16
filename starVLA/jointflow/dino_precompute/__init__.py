"""DINO feature precompute (offline latents).

中文注释：把视频帧过 frozen DINOv3 提 patch tokens，按 dataset/episode 落盘成离线 latents。
训练时 online_dino=auto 即可直接吃这些 latents，免在线提特征。
"""
