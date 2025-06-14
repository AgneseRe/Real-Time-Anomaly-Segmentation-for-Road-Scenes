# BiSeNet Training Resources

This folder contains only some of the files generated during the **BiSeNet** training process, including:

* **automated_log.txt**: Plain text file that contains in columns the following info of each epoch {Epoch, Train-loss,Test-loss,Train-IoU,Test-IoU, learningRate}.
* **best.txt**: Plain text file containing a line with the best IoU achieved during training and its epoch.
* **bisenet.py**: copy of the model file used. 
* **model_best.pth**: saved weights of the epoch that achieved best val accuracy.
* **model.txt**: Plain text that displays the model's layers.
* **opts.txt**: Plain text file containing the options used for this training.

> ⚠️ Due to the large size of certain files — some of them exceed GitHub’s file size limitations — they have been stored externally on Google Drive to keep the repository lightweight and manageable. In addition to the previous files, in the Drive you can also find **checkpoint.pth.tar** that contains the checkpoint of the last trained epoch, and **model_best.pth.tar** that contains the same parameters as "checkpoint.pth.tar" but for the epoch with best val accuracy.

## 🔗 Access the Complete Training Materials

[**Click here to open the Google Drive folder**](https://drive.google.com/drive/folders/1fAm8san3WTsCS3yB08mgxdaGFYr1AscU?usp=sharing)

---

Please ensure you have the necessary permissions to view and download the files.  
If you encounter any issues accessing the folder, feel free to contact the repository maintainer.