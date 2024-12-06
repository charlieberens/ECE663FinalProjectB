import pickle

import torch

def main():
    # Open temp.pth file
    model_state_dict = torch.load("temp.pth")  

    # Print the data
    # obj_of_the_things = {
    #     "compression_distances": compression_distances,
    #     "noise_distances": noise_distances,
    #     "different_distances": different_distances
    # }
    compression_distances = model_state_dict["compression_distances"]
    noise_distances = [*model_state_dict["noise_distances"]]
    different_distances = model_state_dict["different_distances"]

    # Get 5 minimum different distances
    different_distances.sort()
    min_different_distances = different_distances[:5]
    print(f"5 minimum different distances: {min_different_distances}")

    thresh = .000175

    # Get percentage of noise distances that are less than thresh
    # Flatten noise_distances
    noise_distances = [dist for sublist in noise_distances for dist in sublist]
    num_noise_distances = len(noise_distances)
    num_noise_distances_less_than_thresh = len([dist for dist in noise_distances if dist < thresh])
    percent_noise_distances_less_than_thresh = num_noise_distances_less_than_thresh / num_noise_distances
    print(f"Percentage of noise distances less than {thresh}: {percent_noise_distances_less_than_thresh}")

    # Get percentage of compression distances that are less than thresh
    # Flatten compression_distances
    compression_distances = [dist for sublist in compression_distances for dist in sublist]
    num_compression_distances = len(compression_distances)
    num_compression_distances_less_than_thresh = len([dist for dist in compression_distances if dist < thresh])
    percent_compression_distances_less_than_thresh = num_compression_distances_less_than_thresh / num_compression_distances
    print(f"Percentage of compression distances less than {thresh}: {percent_compression_distances_less_than_thresh}")

if __name__ == '__main__':
    main()