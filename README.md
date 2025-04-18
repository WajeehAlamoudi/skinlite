# 🔸 Step 2.1: show sample
transformed_image, label = train_dataset.__getitem__(7)
print(f'val dataset samples: {train_dataset.__len__()}')
# Save
transformed_save_path = os.path.join(run_dir, f"transformed_label{label}.jpg")
to_pil_image(transformed_image).save(transformed_save_path)
print(f"✅ Saved transformed to: {transformed_save_path}")