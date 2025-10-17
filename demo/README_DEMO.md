test-------------------------------------------


python demo/check_model.py predict --model notebooks/models/baseline_cnn_final_20251016-152849.keras --image assets/test_images/Open_Eyes/47.jpg



python demo/check_model.py predict --model notebooks/models/cnn_lstm_final_20251016-153523_cnnlstm.keras --image assets/test_images/Open_Eyes/47.jpg



python demo/check_model.py predict --model notebooks/models/efficientnet_final_20251016-155501_efficientnet.keras --image assets/test_images/Open_Eyes/47.jpg


python demo/check_model.py predict --model notebooks/models/vit_final_20251016-160528_vit.keras --image assets/test_images/Open_Eyes/14.jpg




python demo/check_model.py predict --model notebooks/models/transfer_mobilenetv2_final_20251016-154036_mobilenetv2.keras --image assets/test_images/Open_Eyes/47.jpg








probe a model-----------------------------

python demo/check_model.py probe --model notebooks/models/baseline_cnn_final_20251016-152849.keras


python demo/check_model.py probe --model notebooks/models/vit_final_20251016-160528_vit.keras

python demo/check_model.py probe --model notebooks/models/efficientnet_final_20251016-155501_efficientnet.keras


python demo/check_model.py probe --model notebooks/models/cnn_lstm_final_20251016-153523_cnnlstm.keras


python demo/check_model.py probe --model notebooks/models/transfer_mobilenetv2_final_20251016-154036_mobilenetv2.keras