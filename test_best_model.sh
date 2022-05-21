# python src/utils/train_test_split.py 
python train.py "general.experiment_name=best_model"
python create_submission.py best_model
cp outputs/best_model/submission.csv submission.csv
cp outputs/best_model/probabilities.csv probabilities.csv