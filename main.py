from src.model_evaluation import ModelEvaluation


if __name__ == "__main__":
    model_evaluation_exec = ModelEvaluation()
    model_evaluation_exec.evaluate_cv()
