import random

optimizers = [
    ("adagrad", 217, 14),
    ("adam", 130, 16),
    ("adamax", 217, 15),
    ("ftrl", 435, 16),
    ("nadam", 217, 15),
    ("sgd", 174, 16),
    ("rmsprop", 130, 16),
]

header = "optimizer,cpu,memory,time,epochs,predictions,dataset,loss,accuracy\n"

def generate_data(optimizer, predictions, dataset):
    cpu = round(predictions / 10 + random.uniform(-0.5, 0.5), 4)
    memory = round(dataset * 1.1 + random.uniform(-0.5, 0.5), 4)
    time = round(dataset * 1.5 + random.uniform(-0.5, 0.5), 4)
    epochs = random.randint(8, 16)
    loss = round((1 - (dataset / 20)) + random.uniform(-0.01, 0.01), 5)
    accuracy = round(dataset / 20 + random.uniform(-0.01, 0.01), 5)

    return f"{optimizer},{cpu},{memory},{time},{epochs},{predictions},{dataset},{loss},{accuracy}\n"

def generate_dataset(repeats=100):
    # dataset = header
    dataset = ""

    for _ in range(repeats):
        for optimizer, predictions, dataset_value in optimizers:
            dataset += generate_data(optimizer, predictions, dataset_value)

    return dataset

def save_to_csv(file_name, data):
    with open(file_name, "w") as file:
        file.write(data)

new_dataset = generate_dataset()
print(new_dataset)
save_to_csv("data/generated_dataset.csv", new_dataset)