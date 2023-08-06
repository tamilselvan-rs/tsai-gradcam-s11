# tsai-session-11

## Folder Structure

```
.  
|__ models
    |__ ResNet.py # RestNet18 and ResNet34 Models
|__ CifarDataSet  # Custom Class to enable Albumen Transforms
|__ utils
    |__ common    # Runner Configuration
    |__ plots     # Matplot Helpers
    |__ modelhelper # Train / Test and Misclassification Helper
    |__ transforms # Train / Test and Inverse Transforms
    |__ visualize  # Dataset Visualizer (Pre and Post Transforms) 
    |__ dataloader.py   # Utils related to Data Loading (batchsize and data source selection)  
|__ *.ipynb         # Executions
```

## Model Output
[GradCam.ipynb](./GradCam.ipynb)

## Image Miss-classifications
![Miss-Class-Images](./media/Screenshot%202023-08-05%20at%2011.22.12%20PM.png)

## Gradcam Output

![Gradcam-1](./media/Screenshot%202023-08-05%20at%2011.21.33%20PM.png)
![Gradcam-2](./media/Screenshot%202023-08-05%20at%2011.22.02%20PM.png)


## Important Resources

[ResNet Implementation](https://github.com/kuangliu/pytorch-cifar)  
[Grad-Cam Library](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html)  
[How to save an entire model?](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
