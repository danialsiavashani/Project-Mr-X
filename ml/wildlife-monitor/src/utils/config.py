NUM_CLASSES = 15
IMG_SIZE = 224
BATCH_SIZE = 32

SEED = 42

# Must match ImageFolder's alphabetical ordering exactly
CLASS_NAMES = [
    'american_crow', 'anna_hummingbird', 'black_phoebe', 'bushtit',
    'california_quail', 'california_scrub_jay', 'cat', 'dog',
    'eurasian_collared_dove', 'house_finch', 'mourning_dove',
    'northern_mockingbird', 'song_sparrow', 'squirrel', 'white_crowned_sparrow'
]

EFFICIENTNET_STAGE1_EPOCHS = 5
EFFICIENTNET_STAGE2_EPOCHS = 10
EFFICIENTNET_STAGE1_LR     = 0.001
EFFICIENTNET_STAGE2_LR     = 0.0001