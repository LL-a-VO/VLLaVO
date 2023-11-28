import Levenshtein

def most_similar_item(word, item_list):
    min_distance = float('inf')
    most_similar_item = None

    for item in item_list:
        distance = Levenshtein.distance(word, item)
        if distance < min_distance:
            min_distance = distance
            most_similar_item = item

    return most_similar_item

def get_dataset_classes(name):
    if name == "Office31":
        classes = ['back pack', 'bike', 'bike helmet', 'bookcase', 'bottle', 'calculator', 'desk chair', 'desk lamp',
                   'desktop computer', 'file cabinet', 'headphones', 'keyboard', 'laptop computer', 'letter tray',
                   'mobile phone', 'monitor', 'mouse', 'mug', 'paper notebook', 'pen', 'phone', 'printer', 'projector',
                   'punchers', 'ring binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape dispenser', 'trash can']
    elif name == "OfficeHome":
        classes = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
                   'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush',
                   'Couch', 'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen',
                   'Monitor', 'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip',
                   'Batteries', 'Radio', 'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker',
                   'Eraser', 'Bucket', 'Chair', 'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles',
                   'Clipboards', 'Scissors', 'TV', 'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven',
                   'Refrigerator', 'Marker']
    elif name == "VisDA":
        classes = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant',
                   'skateboard', 'train', 'truck']
    elif name == "PACS" :
        classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    elif name == "VLCS" :
        classes = ['bird', 'car', 'chair', 'dog', 'person']
    elif name == "DomainNet":
        classes = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant',
                   'anvil',
                   'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball',
                   'baseball_bat',
                   'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench',
                   'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book', 'boomerang',
                   'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket',
                   'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel',
                   'camera',
                   'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat',
                   'ceiling_fan',
                   'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud',
                   'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon',
                   'crocodile',
                   'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut',
                   'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant',
                   'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger',
                   'fire_hydrant',
                   'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops', 'floor_lamp', 'flower',
                   'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe', 'goatee',
                   'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones',
                   'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital',
                   'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane',
                   'ice_cream',
                   'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop',
                   'leaf',
                   'leg', 'light_bulb', 'lighter', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster',
                   'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave',
                   'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug',
                   'mushroom',
                   'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush',
                   'paint_can', 'palm_tree', 'panda', 'pants', 'paper_clip', 'parachute', 'parrot', 'passport',
                   'peanut',
                   'pear', 'peas', 'pencil', 'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow',
                   'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato',
                   'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control',
                   'rhinoceros', 'rifle', 'river', 'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw',
                   'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark',
                   'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
                   'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock',
                   'speedboat',
                   'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak',
                   'stereo',
                   'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean',
                   'submarine',
                   'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 'table', 'teapot',
                   'teddy-bear',
                   'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 'The_Great_Wall_of_China',
                   'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado',
                   'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 't-shirt',
                   'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide',
                   'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
    elif name == "subDomainNet":
        classes = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird']
    return classes



def get_dataset_domain_name(name):
    domain_name = None
    if name == "Office31":
        domain_name = []
    elif name == "OfficeHome":
        domain_name = ['Art','Clipart','Product','Real World']
    elif name == "VisDA":
        domain_name = []
    elif name == "PACS" :
        domain_name = ['art','photo','cartoon','sketch']
    elif name == "VLCS" :
        domain_name = []
    elif name == "DomainNet":
        domain_name = []

    return domain_name


