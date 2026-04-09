This repository contains code related to the construction and validation of binary classifiers used for anomaly detection among objects in ZTF DR23. 
The classifiers are trained and applied using the dr23-features feature set, which is provided [here](https://snad.space/features/).

`rb_clf_V2`
A real–bogus classifier trained on the updated AKB database.

`sn_clf`
A supernova vs. non-supernova classifier. The positive class consists of 674 spectroscopically confirmed supernovae from the ZTF Bright Transient Survey (BTS), selected after visual inspection. The negative class includes 10,000 randomly selected objects from ZTF DR23, as well as all objects from the AKB database excluding those labeled as supernovae. More details are described in corresponding [directory](https://github.com/snad-space/snad-binary-classifiers/tree/master/sn_clf).

`ztf_field_map`
Code for constructing a sky map showing the coverage of ZTF fields.

`TNS_submit`
Code for submitting new objects to the Transient Name Server (TNS) using the SNAD bot.
