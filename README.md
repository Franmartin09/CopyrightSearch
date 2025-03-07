# CopyrightSearch

CopyrightSearch is a fine-tuned small language model designed to identify copyright statements, authors, and contributions in source C and header files.

## Dataset Overview
- **Total source files (.c / .h):** 250.505
- **Training Process:**
  1. **First phase:** Training with 48.201 samples
  2. **Second phase:** Expanding to 150.000 samples
  3. **Final phase:** Training with the full dataset (250.505 samples)

## Retraining Strategy
- The model will be iteratively retrained with increasing dataset sizes to improve accuracy and generalization.

## Future Improvements
- Optimize model performance and inference speed.
- Enhance detection accuracy with additional fine-tuning steps.
- Explore potential integrations with external AI frameworks.

### License
This project follows an open-source approach. Ensure compliance with relevant licenses when using the dataset and trained models.

---
For any questions or contributions, feel free to reach out!# CopyrightSearch


on APPEND_DATA.JSON there are 9508 files without comments search(' "comments": "" ')