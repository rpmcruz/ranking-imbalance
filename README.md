# Pairwise Scoring Ranking for Class Imbalance

This repository contains some of our work at addressing class imbalance by turning a pairwise scoring ranking into a classifier.

Class imbalance is pervasive in certain classification domains. That is, the class distribution is not uniform, in some cases in an extreme fashion. This is a common problem in medicine and health care where there is a wide dispersion of patients suffering from different disease severities; it is inherent in fraud and fault detection where the anomaly is rare; and in many other fields.

Pairwise scoring ranking are appropriate for class imbalance because:

– In **pairwise** ranking, observations are trained in pairs, which means there is no imbalance during training, within a binary context;
– In **scoring** ranking, predictions are produced individually in the form of a score, making it possible to use them for classification.

Our publications:

1. Cruz, R., Fernandes, K., Cardoso, J. S., & Costa, J. F. P. (2016, July). Tackling class imbalance with ranking. In *Neural Networks (IJCNN), 2016 International Joint Conference on* (pp. 2182-2187). IEEE. [[article]](http://ieeexplore.ieee.org/abstract/document/7727469/)
2. Cruz, R., Fernandes, K., Costa, J. F. P., Ortiz, M. P., & Cardoso, J. S. (2017, June). Ordinal Class Imbalance with Ranking. In *Iberian Conference on Pattern Recognition and Image Analysis* (pp. 3-12). Springer, Cham.
3. Cruz, R., Fernandes, K., Costa, J. F. P., Ortiz, M. P., & Cardoso, J. S. (2017, June). Combining Ranking with Traditional Methods for Ordinal Class Imbalance. In *International Work-Conference on Artificial Neural Networks* (pp. 538-548). Springer, Cham.
4. Pérez-Ortiz, M., Fernandes, K., Cruz, R., Cardoso, J. S., Briceño, J., & Hervás-Martínez, C. (2017, June). Fine-to-Coarse Ranking in Ordinal and Imbalanced Domains: An Application to Liver Transplantation. In *International Work-Conference on Artificial Neural Networks* (pp. 525-537). Springer, Cham. (pp. 2182-2187). IEEE.
5. **[in review]** Cruz, R., Fernandes, K., Costa, J. F. P., Ortiz, M. P., & Cardoso, J. S.. Binary Ranking for Ordinal Class Imbalance.
