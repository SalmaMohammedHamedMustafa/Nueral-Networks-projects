import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import register_keras_serializable
import os
import matplotlib.pyplot as plt


@register_keras_serializable()
class Generator(tf.keras.Model):
    def __init__(self, noise_dim=NOISE_DIM, num_classes=10, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # Increase label embedding dimension
        self.label_embedding = tf.keras.layers.Embedding(num_classes, 100)  # Increased from 50
        self.label_dense = tf.keras.layers.Dense(100, use_bias=False)       # Increased from 50
        
        # Combined noise and label processing
        self.combined_dense = tf.keras.layers.Dense(7*7*256, use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(0.3)  # Added dropout layer

        
        # Create label condition vectors for each layer
        self.cond1 = tf.keras.layers.Dense(128, use_bias=False)  # For first upsampling block
        self.cond2 = tf.keras.layers.Dense(64, use_bias=False)   # For second upsampling block
        
        # Reshape layer
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))
        
        # First upsampling block
        self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.3)  # Added dropout layer
        # Second upsampling block
        self.conv_transpose2 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        self.dropout3 = tf.keras.layers.Dropout(0.3)  # Added dropout layer
        # Output layer with tanh activation
        self.output_layer = tf.keras.layers.Conv2D(1, (5, 5), padding='same', use_bias=True, activation='tanh')
        
        # Define conditioning layers for each block
        self.gamma1_layer = tf.keras.layers.Dense(128)  # For first block conditioning
        self.beta1_layer = tf.keras.layers.Dense(128)
        self.gamma2_layer = tf.keras.layers.Dense(64)   # For second block conditioning
        self.beta2_layer = tf.keras.layers.Dense(64)

    def call(self, inputs, training=False):
        noise, labels = inputs
        
        # Process labels - enhanced embedding
        label_embedding = self.label_embedding(labels)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)
        label_info = self.label_dense(label_embedding)
        
        # Prepare conditional vectors for each layer
        cond1_info = self.cond1(label_embedding)  # For first block
        cond2_info = self.cond2(label_embedding)  # For second block
        
        # Concatenate noise and label info
        x = tf.concat([noise, label_info], axis=1)
        
        # Process combined input
        x = self.combined_dense(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.dropout(x, training=training)  # Apply dropout
        # Reshape
        x = self.reshape(x)
        
        # First upsampling block with conditioning
        x = self.conv_transpose1(x)
        # Apply conditional batch normalization with predefined layers
        x = self.conditional_bn(x, cond1_info, training, self.bn2, self.gamma1_layer, self.beta1_layer)
        x = self.relu2(x)
        x = self.dropout2(x, training=training)
        
        # Second upsampling block with conditioning
        x = self.conv_transpose2(x)
        # Apply conditional batch normalization with predefined layers
        x = self.conditional_bn(x, cond2_info, training, self.bn3, self.gamma2_layer, self.beta2_layer)
        x = self.relu3(x)
        x = self.dropout3(x, training=training)
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def conditional_bn(self, x, condition, training, bn_layer, gamma_layer, beta_layer):
        """Custom conditional batch normalization using predefined layers"""
        # Apply predefined batch normalization
        x = bn_layer(x, training=training)
        
        # Reshape condition for broadcasting
        condition = tf.reshape(condition, [-1, 1, 1, condition.shape[-1]])
        
        # Apply predefined gamma and beta layers
        gamma = gamma_layer(condition)
        beta = beta_layer(condition)
        
        # Apply scale and shift
        return x * (1 + gamma) + beta

    @staticmethod
    def generate_noise(batch_size, noise_dim=NOISE_DIM):
        """Generate Gaussian noise for the generator input."""
        return np.random.normal(0, 1, (batch_size, noise_dim))
    
    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({
            'noise_dim': self.noise_dim,
            'num_classes': self.num_classes
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Extract your custom parameters
        noise_dim = config.pop('noise_dim', 100)  # Default if missing
        num_classes = config.pop('num_classes', 10)  # Default if missing
        
        # Create instance with extracted parameters and remaining config
        return cls(noise_dim=noise_dim, num_classes=num_classes, **config)
    



# Discriminator Class with Label Conditioning
@register_keras_serializable()
class Discriminator(tf.keras.Model):
    def __init__(self, num_classes=10,**kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        # Image processing
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.leaky_relu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        
        self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.leaky_relu2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        
        self.conv3 = tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')
        self.leaky_relu3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        
        # Label processing
        self.label_embedding = tf.keras.layers.Embedding(num_classes, 50)
        self.label_dense = tf.keras.layers.Dense(7*7, use_bias=False)
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(1)  # No activation - using from_logits=True

    def call(self, inputs, training=False):
        images, labels = inputs
        
        # Process images
        x = self.conv1(images)
        x = self.leaky_relu1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x, training=training)
        
        x = self.flatten(x)
        
        # Process labels
        label_embedding = self.label_embedding(labels)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)
        
        # Combine image features and label info
        combined = tf.concat([x, label_embedding], axis=1)
        
        # Output
        return self.output_layer(combined)
    
    def get_config(self):
        config = super(Discriminator, self).get_config()
        config.update({
            'num_classes': self.num_classes
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Extract your custom parameters
        num_classes = config.pop('num_classes', 10)  # Default if missing
        
        # Create instance with extracted parameters and remaining config
        return cls(num_classes=num_classes, **config)


# Auxiliary Classifier to help with conditioning
@register_keras_serializable()
class AuxiliaryClassifier(tf.keras.Model):
    def __init__(self, num_classes=10,**kwargs):
        super(AuxiliaryClassifier, self).__init__(**kwargs)

        # Resize the input from (28, 28, 1) to (32, 32, 1)
        self.resize = tf.keras.layers.Resizing(32, 32)

        # Preprocessing layer to convert 1-channel input to 3 channels
        self.preprocess = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))

        self.num_classes = num_classes

        # Load MobileNetV2 with input shape (32, 32, 3)
        hidden = tf.keras.applications.MobileNetV2(
            input_shape=(32, 32, 3),  # Adjusted to 32x32 with 3 channels
            include_top=False,        # Exclude the top classification layer
            weights='imagenet',       # Use pre-trained ImageNet weights
        )

        # Build the model with resizing, preprocessing, and MobileNetV2
        self.model = tf.keras.Sequential([
            self.resize,                       # Resize to 32x32
            self.preprocess,                   # Convert (32, 32, 1) to (32, 32, 3)
            hidden,                            # MobileNetV2 base
            tf.keras.layers.GlobalAveragePooling2D(),  # Reduce spatial dimensions
            tf.keras.layers.Dense(512, activation='relu'),  # Dense layer
            tf.keras.layers.Dropout(0.5),             # Regularization
            tf.keras.layers.Dense(self.num_classes)  # Output layer
        ])

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
    
    def get_config(self):
        config = super(AuxiliaryClassifier, self).get_config()
        config.update({
            'num_classes': self.num_classes
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Extract your custom parameters
        num_classes = config.pop('num_classes', 10)  # Default if missing
        
        # Create instance with extracted parameters and remaining config
        return cls(num_classes=num_classes, **config)

   
# Modified DCGAN Class with Auxiliary Classification Loss
@register_keras_serializable()
class DCGAN(tf.keras.Model):
    def __init__(self, noise_dim=NOISE_DIM, num_classes=10):
        super(DCGAN, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes

        self.generator = Generator(noise_dim, num_classes)
        self.discriminator = Discriminator(num_classes)

        # Add auxiliary classifier
        self.classifier = AuxiliaryClassifier(num_classes)

        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)
        self.classifier_optimizer = tf.keras.optimizers.Adam(0.0001)

    def compile(self):
        super(DCGAN, self).compile()

    def pretrain_classifier(self, images, labels, epochs=20):
        """Pre-train the classifier on real data"""
        print("Pre-training classifier...")

        # Create dataset from images and labels
        train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        train_dataset = train_dataset.shuffle(len(images)).batch(128)

        # Loss function
        self.classifier.compile(
            optimizer=self.classifier_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        # Train the classifier
        self.classifier.fit(train_dataset, epochs=epochs)
        print("Classifier pre-training complete.")

        # print("Classifier Evaluation")
        # pred = self.classifier(x_test)
        # pred = np.argmax(pred,axis=1)
        # print(accuracy_score(y_test,pred))

    @tf.function
    def train_step(self, data, epoch, total_epochs):
        """Enhanced train step with auxiliary classification loss"""
        real_images, real_labels = data
        batch_size = tf.shape(real_images)[0]

        # Generate noise for fake images
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = self.generator([noise, real_labels], training=True)

            # Get discriminator predictions
            real_output = self.discriminator([real_images, real_labels], training=True)
            fake_output = self.discriminator([generated_images, real_labels], training=True)

            # Get classifier predictions for generated images
            gen_class_logits = self.classifier(generated_images, training=False)

            # Calculate standard GAN losses
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            # Calculate classification loss for generated images
            classification_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)(real_labels, gen_class_logits)

            # Add classification loss to generator loss (weighted)
            lambda_cls = 0.1 + (tf.cast(epoch, tf.float32) / total_epochs) * 0.3
            gen_total_loss = gen_loss + lambda_cls * classification_loss

        # Calculate gradients
        gradients_of_generator = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {
            'd_loss': disc_loss,
            'g_loss': gen_loss,
            'cls_loss': classification_loss
        }

    def discriminator_loss(self, real_output, fake_output):
        # Use from_logits=True for numerical stability
        real_labels = tf.ones_like(real_output) * 0.9  # Label smoothing
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            real_labels, real_output)
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.ones_like(fake_output), fake_output)

    def train(self, images, labels=None, epochs=10, batch_size=128,train_classifier=True):
        """
        Train the model with separate images and labels

        Parameters:
        - images: The training images
        - labels: The corresponding class labels (required)
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        """
        if labels is None:
            raise ValueError("Labels must be provided for conditional GAN training")

        # First pre-train the classifier
        if train_classifier:
            self.pretrain_classifier(images, labels)

        # Create dataset from images and labels
        train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        train_dataset = train_dataset.shuffle(len(images)).batch(batch_size)

        for epoch in range(epochs):
            d_loss_total, g_loss_total, cls_loss_total = 0, 0, 0
            num_batches = 0

            # Iterate through dataset
            for real_images, real_labels in train_dataset:
                # Rescale images to [-1, 1] if needed
                if tf.reduce_max(real_images) > 1.0 or tf.reduce_min(real_images) < -1.0:
                    real_images = tf.clip_by_value(real_images, 0, 255) / 127.5 - 1
                elif tf.reduce_max(real_images) <= 1.0 and tf.reduce_min(real_images) >= 0:
                    real_images = real_images * 2 - 1

                # Train on this batch
                losses = self.train_step((real_images, real_labels), epoch, epochs)
                d_loss_total += losses['d_loss']
                g_loss_total += losses['g_loss']
                cls_loss_total += losses['cls_loss']
                num_batches += 1

            # Calculate average losses
            d_loss_avg = d_loss_total / num_batches
            g_loss_avg = g_loss_total / num_batches
            cls_loss_avg = cls_loss_total / num_batches

            # Print progress
            if epoch % 2 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss_avg:.4f}, G Loss: {g_loss_avg:.4f}, Class Loss: {cls_loss_avg:.4f}")
                self.save_generated_images(epoch)
                self.evaluate_class_accuracy(epoch)

    def evaluate_class_accuracy(self, epoch, num_samples=500):
        """Evaluate how well the generated images match their class labels"""
        # Generate images for each class
        synthetic_images, synthetic_labels = self.generate_synthetic_data(num_samples)

        # Convert images to tensor
        synthetic_images = tf.convert_to_tensor(synthetic_images, dtype=tf.float32)
        synthetic_labels = tf.convert_to_tensor(synthetic_labels, dtype=tf.int32)

        # Get classifier predictions
        predictions = self.classifier(synthetic_images, training=False)
        predicted_classes = tf.argmax(predictions, axis=1).numpy()

        # Calculate accuracy
        accuracy = np.mean(predicted_classes == synthetic_labels.numpy())
        print(f"Epoch {epoch + 1} - Generated Image Class Accuracy: {accuracy:.4f}")

        # Create confusion matrix
        confusion = tf.math.confusion_matrix(
            labels=synthetic_labels,
            predictions=predicted_classes,
            num_classes=self.num_classes
        ).numpy()

        # Print per-class accuracy
        class_accuracies = np.diag(confusion) / np.sum(confusion, axis=1)
        for i, acc in enumerate(class_accuracies):
            print(f"  Class {i} accuracy: {acc:.4f}")

    

    def generate_synthetic_data(self, num_samples, num_classes=10):
        """
        Generate synthetic data with classifier verification.

        Args:
            num_samples (int): Total number of synthetic examples to generate.
            num_classes (int): Number of classes (digits 0-9).
            batch_size (int): Number of images to generate per batch.

        Returns:
            np.array: Synthetic images and their verified labels.
        """
        if num_classes is None:
            num_classes = self.num_classes

        synthetic_images, synthetic_labels = [], []
        samples_per_class = num_samples // num_classes

        for digit in range(num_classes):
            # Create noise and labels
            noise = tf.random.normal([samples_per_class, self.noise_dim])
            labels = tf.ones(samples_per_class, dtype=tf.int32) * digit

            # Generate class-specific images
            generated_imgs = self.generator([noise, labels], training=False)
            generated_imgs = (generated_imgs + 1) / 2.0  # Rescale to [0, 1]

            synthetic_images.extend(generated_imgs.numpy())
            synthetic_labels.extend([digit] * samples_per_class)

        return np.array(synthetic_images), np.array(synthetic_labels)

    def save_generated_images(self, epoch, output_dir='generated_images'):
        """Save generated images for each class for visualization."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate one image per class in a 3x4 grid (for 10 classes)
        fig, axs = plt.subplots(3, 4, figsize=(12, 9))
        axs = axs.flatten()

        # Generate sample for each class
        for digit in range(min(10, self.num_classes)):
            noise = tf.random.normal([1, self.noise_dim])
            label = tf.constant([digit], dtype=tf.int32)
            gen_img = self.generator([noise, label], training=False)
            gen_img = (gen_img + 1) / 2.0  # Rescale to [0, 1]

            axs[digit].imshow(gen_img[0, :, :, 0], cmap='gray')
            axs[digit].set_title(f"Class {digit}")
            axs[digit].axis('off')

        # Remove any unused subplots
        for i in range(self.num_classes, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'epoch_{epoch}_classes.png'))
        plt.close()

    def _ensure_optimizers_initialized(self):
        if self.generator.trainable_variables:
            zero_grads_g = [tf.zeros_like(v) for v in self.generator.trainable_variables]
            self.generator_optimizer.apply_gradients(zip(zero_grads_g, self.generator.trainable_variables))
        if self.discriminator.trainable_variables:
            zero_grads_d = [tf.zeros_like(v) for v in self.discriminator.trainable_variables]
            self.discriminator_optimizer.apply_gradients(zip(zero_grads_d, self.discriminator.trainable_variables))
        if self.classifier.trainable_variables:
            zero_grads_c = [tf.zeros_like(v) for v in self.classifier.trainable_variables]
            self.classifier_optimizer.apply_gradients(zip(zero_grads_c, self.classifier.trainable_variables))

    def save(self, checkpoint_dir):
        self.generator.save(os.path.join(checkpoint_dir, 'generator.keras'))
        self.discriminator.save(os.path.join(checkpoint_dir, 'discriminator.keras'))
        self.classifier.save(os.path.join(checkpoint_dir, 'classifier.keras'))
        self._ensure_optimizers_initialized()
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            classifier_optimizer=self.classifier_optimizer
        )
        checkpoint.save(os.path.join(checkpoint_dir, 'optimizer_checkpoint'))
        # Save optimizer states
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, os.path.join(checkpoint_dir, 'optimizer_ckpt'), max_to_keep=1
        )
        checkpoint_manager.save()
        
        # Save additional configurations
        config = {
            'noise_dim': self.noise_dim,
            'num_classes': self.num_classes
        }
        np.save(os.path.join(checkpoint_dir, 'config.npy'), config)
        
        print(f"Model saved successfully to {checkpoint_dir}")
    
 
    @classmethod
    def load(cls, checkpoint_dir='checkpoints', custom_objects=None):
        """Load a complete GAN model from a checkpoint directory"""
        # Load configuration
        config_path = os.path.join(checkpoint_dir, 'config.npy')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        config = np.load(config_path, allow_pickle=True).item()
        
        # Create a new instance with the saved configuration
        gan = cls(noise_dim=config['noise_dim'], num_classes=config['num_classes'])
        
        # Initialize the models
        gan._initialize_models()
        
        # Load models
        gan.generator = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, 'generator.keras'),
            custom_objects=custom_objects
        )
        
        gan.discriminator = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, 'discriminator.keras'),
            custom_objects=custom_objects
        )
        
        gan.classifier = tf.keras.models.load_model(
            os.path.join(checkpoint_dir, 'classifier.keras'),
            custom_objects=custom_objects
        )
        
        # Initialize optimizers
        gan._ensure_optimizers_initialized()
        
        # Create optimizer checkpoint
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=gan.generator_optimizer,
            discriminator_optimizer=gan.discriminator_optimizer,
            classifier_optimizer=gan.classifier_optimizer
        )
        
        # Find latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(
            os.path.join(checkpoint_dir, 'optimizer_ckpt')
        )
        
        if latest_checkpoint:
            # Restore optimizer states
            checkpoint.restore(latest_checkpoint)
            print(f"Optimizer states restored from {latest_checkpoint}")
        else:
            print("Warning: No optimizer checkpoint found. Starting with fresh optimizers.")
        
        print(f"Model loaded successfully from {checkpoint_dir}")
        return gan
    
    def _initialize_models(self):
        """Initialize models by running a dummy forward pass"""
        # Create dummy data
        batch_size = 2
        dummy_images = tf.random.normal([batch_size, 28, 28, 1])
        dummy_labels = tf.random.uniform([batch_size], 0, self.num_classes, dtype=tf.int32)
        
        # Forward pass through each model to build them
        noise = tf.random.normal([batch_size, self.noise_dim])
        _ = self.generator([noise, dummy_labels], training=False)
        _ = self.discriminator([dummy_images, dummy_labels], training=False)
        _ = self.classifier(dummy_images, training=False)
