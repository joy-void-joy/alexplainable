import pathlib
import csv


def most_common(lst):
    return max(set(lst), key=lst.count)


csv_path = pathlib.Path("./data/imagenet/LOC_val_solution.csv")
val_path = pathlib.Path("./data/imagenet/ILSVRC/Data/CLS-LOC/val")

solution = list(csv.DictReader(csv_path.read_text().splitlines(), delimiter=","))

for i in solution:
    move_to = val_path / most_common(i["PredictionString"].split()[::5])
    move_to.mkdir(exist_ok=True, parents=True)

    file = val_path / (i["ImageId"] + ".JPEG")
    file.rename(move_to / file.name)
