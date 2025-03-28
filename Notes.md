Notes about the code:

# Difference between train and train_mp

- Both use the same argmuents
- train sets device to cuda:0, while train_mp gets the local rank environment variable and uses torch.cuda.set_device(args.local_rank). This is supposed to find available devices and set them as the script's devices, but it only really works when the script has all GPUs in the node

--------------------------- Same as GCD ----------------------------------------

- Both script then run the init_experiment function, which creates logging and checkpoint saving directory and assigns an experiment ID from the date if none was already given. I also sets up a loguru logger for the script.
- Now both scripts set up the model, which is VIT-b pre-trained with Dino, and freeze all the layers except the last block.
- Then it creates the augmentation transformer for the train and test data, and it creates an augmenter to generate multiple views of the training data for contrastive learning
- Then it loads the datasets with these transforms, and creates a sampler to balance labelled and unlabelled classes. The sampler assigns 1 weight to labeled data and len(all_data)/len(unlabeled_data) weight to unlabeled data. Train_mp uses a distributed sampler
- Then the dataloaders are made and the prediction head is created. I have no idea why the prediction head is made alone, and I should **invistigate that further**

--------------------------- My GCD notes end here ------------------------------

- Both split the model parameters into regularized and non-regularized groups. Bias and norm paramters are not to be regularized, and that is done by giving them 0 weight decays in the optimizer.
- Both create an SGD optimizer. Difference being that train_mp scales the learning rate by the batch size and amount of GPUs.
- Both enable support for Automatic Mixed Precision (AMP), using a floating point 16 scaler and autocast if requested by arguemnts
- Both carry the same training, where they pass images through the model and compute the different losses with, then they compute backprop and optimizer step and scheduler step.
- Both models perform testing on the unlabeled classes in trainset and save model with its optimizer and epoch number.

# Understanding the dataset
## Contrastive views and transform
Trainset transforms:
- Resize (Dictated by crop_pct and interpolation)
- Random crop
- Random horizontal flip (50% chance)
- ColorJitter
- Normalize following imagenet values (I think) and transform to tensor
- Multiple view generation for contrastive loss. Meaning in the end one image becomes multiple after transform

Testset transforms:
- Resize (Dictated by crop_pct and interpolation)
- Center crop
- Normalize following imagenet values (I think) and transform to tensor

Contrastive views: Takes an image and creates a list of it being transformed n_view times. Used for self-supervised learning.

## The dataset itself
A typical dataset maker creates a train_set then splits it into labelled and unlabelled datasets using prop_train_labels as splitting percentage (usually 50%)

Then it takes the labelled trainset and makes a validation set out of it if needed (Default is False)

Finally it also provides a test set too. Giving overall a labelled and unlabelled trainset, a labelled valset (if applicable otherwise None) and a full testset.

Of course trainset and testset are given their respective transform for the samples, but all datasets are assigned a target transform that maps the classes into a list of indices.

In the main get_dataset function the labelled and unlabelled trainsets are merged, so that a dataloader can get both labelled and unlabelled samples.

Then the unlabelled trainset is also copied and given the test transform instead. This dataset can be used for testing (or verifying) the performance on unlabelled data.

The provided datasets are the following:
- Merged trainset
- Testset
- Unlabelled trainset for testing
- Original datasets given by the inner function

A normal dataset provides the image and label and a uq_idx that is not used and I do not understand. The merged dataset provides a value denoting if the given data is labelled or not.

# Understanding the model
To be honest there doesn't seem to be anything special in the model. I think I might not need to change anything, except maybe modifying the head to do hyperbolic stuff. Although if I choose to map after the head's output I wouldn't need to do that.

The model outputs two values:
- The hidden embedding before being passed through the last layer (Used for contrastive learning)
- The logit values from the last layer (Used by clustering loss)

# Understanding the training/losses

Training data has the shape (B,V,H,W,C), where V is views and is usually 2. I am not entirely sure if view values other than 2 work, as a lot of the splitting functions are hard coded to split in two. I am not even sure if they split correctly, since no dim is provided.

## Contrastive losses
Supervised:
- Only done on labelled data
- It is normal contrastive loss, but with all features from the same class in the numerator
- In the code it takes features and labels or mask, in case you create the mask beforehand
- Expected feature dimension is (batch size, #views) confirming my theory
- If no mask is provided then a mask is created, as a matrix with 1 values where labels are equal
- If no labels or mask, then the created mask is the identity matrix
- Supports two anchor modes, "one" which uses only one view, and "all" which uses all views. All is done by concatinating the feature from all views before doing dot product
- Computes loss following the formula Idk. I should look into how the code maps to the actual equation. An opportunity to write math on the whiteboard huehuehue

Unsupervised:
- Performs info_nce loss, where it creates similarity matrix by multiplying (dot product?) the features with each other
- Creates a mask over scores of positive samples and uses it to seperate positive and negative samples
- After all this it concatenates the positive and negative scores and then performes cross entropy loss with zeroes (Because we want to minimize distance I guess)


## Clustering losses
Supervised:
- Splits the output of the views and concatenates it after each other, the performs cross entropy loss with the labels, duplicated by the number of views.
- Of course uses the supervised label mask to only work with labelled data

Unsupervised:
- Paper doesn't really make sense and the code does not really follow what is done in the paper
- Works with a student and teacher predictions, teacher predictions start as a copy of student predictions.
- Views are divided and placed under each other in the batch (.chunk(2))
- Student is multiplied by student temp (Hard coded somewhere to be 0.1)
- Teacher is multiplied by moving temp that goes from 0.07 to 0.04 in a warmup and stays there after warmup
- For some reason teacher uses softmax while student uses log softmax
- All teacher/student pairs (including both views) are cross entropied. Except the views that are identical (same image and view), which are ignored.
- The sum of the cross entropy is normalized and returned. This is cluster_criterion
- After that the entropy is calculated and added with a weight to the loss. I have no idea how they calculate this entropy. It looks weird

# Testing


# Plan
- We need to implement an exponential map to move projections to hyperbolic space
- We need to implement a hyperbolic distance function for contrastive loss
- We need to implement a hyperbolic prediction layer with hyperbolic softmax and hyperbolic cross entropy for parametric clustering

# Notes
I do not really understand the shape of the data