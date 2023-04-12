from src.model import *

def main():
    print("Execute Models......")
    # Executing the sequential model.
    model1 = model_Sequential()

    # Executing the resnet_50 model.
    model2 = resnet_50()

if __name__ == "__main__":
    main()