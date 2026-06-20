# Error Analysis

## Misclassified Samples Review

Five misclassified validation images were saved and reviewed from the 
   ```text 
    reports/errors/
``` 
folder.

A significant portion of the observed errors involved the Pepper Bell Bacterial Spot class. This suggests that the model occasionally struggles to distinguish bacterial spot symptoms from visually similar leaf conditions.

Possible reasons include:

1. Similar lesion appearance across disease categories.
2. Variations in disease severity and infection stages.
3. Lighting and image quality differences.
4. Background noise and leaf orientation changes.

## Classification Report Findings

The model achieved an overall validation accuracy of 93%.

The strongest-performing class was Tomato Healthy, achieving perfect recall (100%), indicating that all healthy tomato leaves were correctly classified.

The most challenging classes were:

* Tomato Early Blight (Recall = 72%)
* Pepper Bell Bacterial Spot (Precision = 90%, Recall = 93%)

These results suggest that disease classes with visually similar spotting patterns remain the primary source of classification errors.

## Conclusion

Most model errors occurred between disease categories that share similar visual symptoms rather than between healthy and diseased leaves. Additional training data, targeted augmentation, or partial fine-tuning of deeper network layers may further reduce these confusions.
