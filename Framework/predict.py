# imports
import numpy as np

def norm(num, off, fac):
    return (num-off)/fac

def rev_norm(num, off, fac):
    return (num*fac) + off

def predict_classification(obj, offset=0, factor=1):
    while True:
        try:
            num = int(input("Enter a value to predict (or press ctrl + C to end process) : "))

            num_arr = np.array([norm(num, offset, factor)]).reshape(-1, 1)
            prediction = obj.predict(num_arr)
            prediction = 1 if prediction>0.5 else 0

            print("predicted value is : {}\n".format(prediction))
            # print("predicted value is : {}\n".format(prediction))

        except KeyboardInterrupt as e:
            print("\n\n####====####====EXITING NEURAL NET====####====####\n")
            break

        except Exception as e:
            # print(e.message)
            print("Wrong Input, try again\n")
            continue

def predict_regression(obj, offset=0, factor=1):
    while True:
        try:
            num = int(input("Enter a value to predict (or press ctrl + C to end process) : "))

            num_arr = np.array([norm(num, offset, factor)]).reshape(-1, 1)
            prediction = obj.predict(num_arr)

            print("predicted value is : {}\n".format(prediction))
            # print("predicted value is : {}\n".format(prediction))

        except KeyboardInterrupt as e:
            print("\n\n####====####====EXITING NEURAL NET====####====####\n")
            break

        except Exception as e:
            # print(e.message)
            print("Wrong Input, try again\n")
            continue
