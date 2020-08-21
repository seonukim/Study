import tensorflow as tf

mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.ReductionToOneDevice())
# model
with mirrored_strategy.scope():
    takemodel = EfficientNetB2(weights='imagenet',include_top = False, input_shape = (256, 256, 3))
    takemodel.trainable = True
    # takemodel.summary()

    layer_dict = dict([(layer.name, layer) for layer in takemodel.layers])

    x = layer_dict['top_activation'].output
    print(x)
    x = Conv2D(filters=300, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='softmax')(x)

    model = Model(takemodel.input, x)
    model.summary()
    model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])    