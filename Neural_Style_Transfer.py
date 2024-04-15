# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="Dc1uzxgNPI0G"
# # Necessary Imports

# %% id="BmH3R0NllQQn" executionInfo={"status": "ok", "timestamp": 1713207182044, "user_tz": -330, "elapsed": 4445, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
import PIL

# %% colab={"base_uri": "https://localhost:8080/"} id="l9FeiDWnlzBR" executionInfo={"status": "ok", "timestamp": 1713207214808, "user_tz": -330, "elapsed": 32767, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}} outputId="2f9ca15c-e892-4cc3-c765-9ae9b29cbec5"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="U8lkiQ6wPBX6"
# # Load VGG-19 Pretrained-Model

# %% id="1B-CJq8NlQQo" executionInfo={"status": "ok", "timestamp": 1713207715374, "user_tz": -330, "elapsed": 6984, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
img_size = 400
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='/content/drive/MyDrive/Colab Notebooks/Neural_Style_Transfer/pretrained-model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

vgg.trainable = False


# %% [markdown] id="TBUbkBpjPNFq"
# # Content Loss
#
# $$J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2 $$

# %% nbgrader={"grade": false, "grade_id": "cell-3d3bfd0678816054", "locked": false, "schema_version": 3, "solution": true, "task": false} id="MFa2QIiWlQQr" executionInfo={"status": "ok", "timestamp": 1713207715375, "user_tz": -330, "elapsed": 18, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])


    J_content =  tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))/(4.0 * n_H * n_W * n_C)


    return J_content


# %% [markdown] id="9XJNsP0mPwoO"
# # Gram Matrix
#
#

# %% nbgrader={"grade": false, "grade_id": "cell-332b0f746ef0069e", "locked": false, "schema_version": 3, "solution": true, "task": false} id="EL2CMJSglQQs" executionInfo={"status": "ok", "timestamp": 1713207715375, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))
    return GA


# %% [markdown] id="fz15rcwtQJFj"
# # Style Cost of one layer
#
# $$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2 $$

# %% nbgrader={"grade": false, "grade_id": "cell-8f37df6f128c1f99", "locked": false, "schema_version": 3, "solution": true, "task": false} id="Secfzu4qlQQt" executionInfo={"status": "ok", "timestamp": 1713207715375, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()


    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))

    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)


    J_style_layer = tf.reduce_sum(tf.square(GS - GG))/(4.0 *(( n_H * n_W * n_C)**2))


    return J_style_layer


# %% [markdown] id="J-1bATcNQYGI"
# # VGG-19 model layers

# %% colab={"base_uri": "https://localhost:8080/"} id="icxjeGzOlQQu" executionInfo={"status": "ok", "timestamp": 1713207715375, "user_tz": -330, "elapsed": 17, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}} outputId="d465068b-14d6-4f27-bcb3-ac7c6083355b"
for layer in vgg.layers:
    print(layer.name)

# %% colab={"base_uri": "https://localhost:8080/"} id="67OxpopVlQQu" executionInfo={"status": "ok", "timestamp": 1713207715375, "user_tz": -330, "elapsed": 6, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}} outputId="3505c3a9-43a3-47b7-e640-1bde34714325"
vgg.get_layer('block5_conv2').output

# %% id="mbPGZoJklQQu" executionInfo={"status": "ok", "timestamp": 1713208533179, "user_tz": -330, "elapsed": 908, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
STYLE_LAYERS = [
    ('block1_conv1', 0.1),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.4),
    ('block4_conv1', 0.1),
    ('block5_conv1', 0.2)]


# %% [markdown] id="I8bgjnmOQggu"
# # Total Style Loss
#
# $$J_{style}(S,G) = \sum_{l} \lambda^{[l]} J^{[l]}_{style}(S,G)$$

# %% deletable=false id="1NJ2kU5olQQz" executionInfo={"status": "ok", "timestamp": 1713207715375, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    style_image_output -- vgg model outputs
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    a_S = style_image_output[1:]


    a_G = generated_image_output[1:]

    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


# %% [markdown] id="qgxscJlUQuv7"
# # Total Cost
#
# $$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$

# %% nbgrader={"grade": false, "grade_id": "cell-55270d5342632932", "locked": false, "schema_version": 3, "solution": true, "task": false} id="SrPrZSYFlQQz" executionInfo={"status": "ok", "timestamp": 1713207715375, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style


    return J

# %% [markdown] id="mm3l9ZJnQ4Ke"
# # Load Content and Style Images

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} id="6tgQpRkxlQQ0" executionInfo={"status": "ok", "timestamp": 1713207717999, "user_tz": -330, "elapsed": 2627, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}} outputId="6d286c6a-3ef4-4cf6-cfd5-5d1d097212d7"
content_image = np.array(Image.open("/content/drive/MyDrive/Colab Notebooks/Neural_Style_Transfer/images/YellowLabradorLooking_new.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

print(content_image.shape)
imshow(content_image[0])
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} id="exyf_lghlQQ1" executionInfo={"status": "ok", "timestamp": 1713207720492, "user_tz": -330, "elapsed": 2498, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}} outputId="437420f5-9990-4c0b-f18a-b15b7bbbfab0"
style_image =  np.array(Image.open("/content/drive/MyDrive/Colab Notebooks/Neural_Style_Transfer/images/Vassily_Kandinsky,_1913_-_Composition_7.jpg").resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

print(style_image.shape)
imshow(style_image[0])
plt.show()

# %% [markdown] id="D7wlWaFdQ8mO"
# # Random generated image

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} id="6lhED1UQlQQ1" executionInfo={"status": "ok", "timestamp": 1713207723734, "user_tz": -330, "elapsed": 3256, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}} outputId="7b97e8a6-9820-4b45-e6f1-8d602f1d13f6"
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.5, 0.5)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

print(generated_image.shape)
imshow(generated_image.numpy()[0])
plt.show()


# %% [markdown] id="ewdtyNiORGQn"
# # Getting activations from VGG-19 of content and style image

# %% id="xOguiM_clQQ1" executionInfo={"status": "ok", "timestamp": 1713207921423, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


# %% id="XZaJ0vNXlQQ1" executionInfo={"status": "ok", "timestamp": 1713208544540, "user_tz": -330, "elapsed": 713, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
content_layer = [('block5_conv2', 1)]

vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

# %% id="6iIS1y1WlQQ2" executionInfo={"status": "ok", "timestamp": 1713207925593, "user_tz": -330, "elapsed": 2721, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style encoder

# %% id="LeVPvCuslQQ2" executionInfo={"status": "ok", "timestamp": 1713207925593, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}

preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

# %% id="Gi8SOwdmlQQ2" executionInfo={"status": "ok", "timestamp": 1713207925594, "user_tz": -330, "elapsed": 4, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)


# %% id="JTweYEHhlQQ2" executionInfo={"status": "ok", "timestamp": 1713207925594, "user_tz": -330, "elapsed": 3, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image

    Arguments:
    tensor -- Tensor

    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


# %% [markdown] id="W2YedOdYRVZG"
# # Train Step

# %% nbgrader={"grade": false, "grade_id": "cell-dfbcc4b8f8a959e5", "locked": false, "schema_version": 3, "solution": true, "task": false} id="fiz25fGhlQQ2" executionInfo={"status": "ok", "timestamp": 1713209205663, "user_tz": -330, "elapsed": 769, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# total_variation_weight = 30

@tf.function()
def train_step(generated_image, a_C, a_S, alpha = 20, beta = 30):
    with tf.GradientTape() as tape:

        a_G = vgg_model_outputs(generated_image)


        J_style = compute_style_cost(a_S, a_G)


        J_content = compute_content_cost(a_C, a_G)

        J = total_cost(J_content, J_style,alpha = alpha, beta = beta)
        # J += total_variation_weight*tf.image.total_variation(generated_image)

    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "output_embedded_package_id": "1a-2olfNE56UOPjUIPOHky8W5IwhMP9am"} id="fp7Ama-qlQQ3" outputId="2dfa614c-9496-431e-e833-2e4e5235792a" executionInfo={"status": "ok", "timestamp": 1713208219652, "user_tz": -330, "elapsed": 163200, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}}
generated_image = tf.Variable(tf.image.convert_image_dtype(generated_image, tf.float32))
epochs = 2501
for i in range(epochs):
    train_step(generated_image, a_C, a_S)
    if i % 250 == 0:
        print(f"Epoch {i} ")
    if i % 250 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        image.save(f"/content/drive/MyDrive/Colab Notebooks/Neural_Style_Transfer/output/image_{i}.jpg")
        plt.show()

# %% [markdown] id="bC_07agHRYnT"
# # Generated image through style transfer

# %% colab={"base_uri": "https://localhost:8080/", "height": 391} id="8TJfEIp3lQQ3" executionInfo={"status": "ok", "timestamp": 1713208274326, "user_tz": -330, "elapsed": 5497, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}} outputId="019970bf-bd12-4b01-b8fd-9392edcc7ed6"
# Show the 3 images in a row
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
imshow(content_image[0])
ax.title.set_text('Content image')
ax = fig.add_subplot(1, 3, 2)
imshow(style_image[0])
ax.title.set_text('Style image')
ax = fig.add_subplot(1, 3, 3)
imshow(generated_image[0])
ax.title.set_text('Generated image')
plt.show()

# %% [markdown] id="bnJc10tkRkMz"
# # Another example

# %% id="G40q6so3_gXF" colab={"base_uri": "https://localhost:8080/", "height": 1000, "output_embedded_package_id": "1vjKUHeHIaUo6RQSnWvth0KJbqpVv9xuv"} executionInfo={"status": "ok", "timestamp": 1713209374476, "user_tz": -330, "elapsed": 163075, "user": {"displayName": "Abhay Vijayvargiya", "userId": "11157537961666642085"}} outputId="32bb99a2-aa4a-49c7-fd39-5db0ceb689b9"
content_image = np.array(Image.open("/content/drive/MyDrive/Colab Notebooks/Neural_Style_Transfer/images/persian_cat.jpg").resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

print(content_image.shape)
imshow(content_image[0])
plt.show()


style_image =  np.array(Image.open("/content/drive/MyDrive/Colab Notebooks/Neural_Style_Transfer/images/stone_style.jpg").resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

print(style_image.shape)
imshow(style_image[0])
plt.show()

generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
# noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
# generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)


preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)


generated_image = tf.Variable(tf.image.convert_image_dtype(generated_image, tf.float32))
epochs = 2501
for i in range(epochs):
    train_step(generated_image, a_C, a_S)
    if i % 250 == 0:
        print(f"Epoch {i} ")
    if i % 250 == 0:
        image = tensor_to_image(generated_image)
        imshow(image)
        image.save(f"/content/drive/MyDrive/Colab Notebooks/Neural_Style_Transfer/output/image_2_{i}.jpg")
        plt.show()


# Show the 3 images in a row
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
imshow(content_image[0])
ax.title.set_text('Content image')
ax = fig.add_subplot(1, 3, 2)
imshow(style_image[0])
ax.title.set_text('Style image')
ax = fig.add_subplot(1, 3, 3)
imshow(generated_image[0])
ax.title.set_text('Generated image')
plt.show()

# %% [markdown] id="irXQ82tjRtAE"
# ## We can tune different hyperparameters(alpha, beta, weights of each layer in style Cost) to change the texture and style.

# %% id="XXwbsivFSzW3"
