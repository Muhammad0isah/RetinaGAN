import os
import argparse
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from models.cstylegan import cStyleGAN
from models.gaugan import GauGAN
from utils import fix_pred_label, onehot_to_rgb, color_dict

parser = argparse.ArgumentParser()
parser.add_argument('--dr_grade', type=int, required=True, help='DR grade (class label) 0–4')
parser.add_argument('--num_samples', type=int, default=10, help='Number of images to generate')
parser.add_argument('--output_dir', type=str, default='generated_images', help='Directory to save generated images')
args = parser.parse_args()

BATCH_SIZE = 4
CLASS_LABEL = args.dr_grade
NUM_SAMPLES = args.num_samples
SAVE_DIR = os.path.join(args.output_dir, str(CLASS_LABEL))
os.makedirs(SAVE_DIR, exist_ok=True)

##### LOAD Conditional StyleGAN
conditional_style_gan = cStyleGAN(start_res=4, target_res=1024)
conditional_style_gan.grow_model(256)
conditional_style_gan.load_weights(os.path.join('checkpoints/cstylegan/cstylegan_256x256.ckpt')).expect_partial()

##### LOAD GauGAN
gaugan = GauGAN(image_size=1024, num_classes=7, batch_size=BATCH_SIZE, latent_dim=512)
gaugan.load_weights('checkpoints/gaugan/gaugan_1024x1024.ckpt')

# Number of full batches
steps = NUM_SAMPLES // BATCH_SIZE
remainder = NUM_SAMPLES % BATCH_SIZE
if remainder != 0:
    steps += 1

count = 0
for step in range(steps):
    current_batch = BATCH_SIZE if (count + BATCH_SIZE <= NUM_SAMPLES) else NUM_SAMPLES - count

    z = tf.random.normal((current_batch, conditional_style_gan.z_dim))
    w = conditional_style_gan.mapping([z, conditional_style_gan.embedding(CLASS_LABEL)])
    noise = conditional_style_gan.generate_noise(batch_size=current_batch)
    labels = conditional_style_gan.call({"style_code": w, "noise": noise, "alpha": 1.0, "class_label": CLASS_LABEL})

    labels = tf.keras.backend.softmax(labels)
    labels = tf.cast(labels > 0.5, dtype=tf.float32)
    labels = tf.image.resize(labels, (1024, 1024), method='nearest')
    fixed_labels = fix_pred_label(labels)

    latent_vector = tf.random.normal(shape=(current_batch, 512), mean=0.0, stddev=2.0)
    fake_images = gaugan.predict([latent_vector, fixed_labels])

    for idx in range(current_batch):
        out_img = (fake_images[idx] * 255).astype('uint8')
        save_path = os.path.join(SAVE_DIR, f'synthetic_{CLASS_LABEL}_{count}.png')
        cv2.imwrite(save_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        count += 1


# command to run the script:
# python run_pipeline.py --dr_grade 2 --num_samples 10 --output_dir generated_images
