import get_data
import preprocess
import feature_engineer
import embeddings
import train_model

if __name__ == "__main__":
    # Data download
    downloader = get_data.GetData()
    downloader.kaggle_api()
    downloader.create_folder()
    downloader.download_zip()
    downloader.extract_zip()
    downloader.rename_final()
    downloader.extract_data()
    
    # Preprocessing
    processor = preprocess.Preprocessor()
    processor.load_data()
    processor.clean_data()
    processor.save_data()
    
    # Feature Engineering
    engineer = feature_engineer.FeatureEngineering()
    engineer.load_data()
    engineer.raw_features()
    engineer.handle_distribution()
    engineer.derived_features()
    engineer.reduce_multicollinearity()
    engineer.save_data()
    
    # Generate Embeddings
    generator = embeddings.GenerateEmbeddings()
    generator.load_data()
    generator.sbert_init()
    generator.sbert_encode()
    generator.save_cache()
    generator.similarity_scores()
    generator.save_data()
    
    # Train Model
    trainer = train_model.ModelTraining()
    trainer.load_data()
    trainer.data_split()
    trainer.random_space()
    trainer.grid_space()
    trainer.model_evaluation()
    trainer.save_model()
