"""Ragas evaluation runner."""
import logging
import yaml
from pathlib import Path
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """Evaluate RAG system using Ragas metrics."""
    
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ]
    
    def load_test_dataset(self, yaml_path: str) -> List[Dict]:
        """
        Load test dataset from YAML file.
        
        Args:
            yaml_path: Path to YAML file with test questions
            
        Returns:
            List of test cases
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return data.get('test_cases', [])
    
    def prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dataset:
        """
        Prepare dataset for Ragas evaluation.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context chunks (list of lists)
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Hugging Face Dataset
        """
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts
        }
        
        if ground_truths:
            data['ground_truth'] = ground_truths
        
        return Dataset.from_dict(data)
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dict:
        """
        Run Ragas evaluation.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context chunks
            ground_truths: Optional ground truth answers
            
        Returns:
            Evaluation results dictionary
        """
        try:
            # Prepare dataset
            dataset = self.prepare_dataset(
                questions,
                answers,
                contexts,
                ground_truths
            )
            
            # Run evaluation
            results = evaluate(dataset, metrics=self.metrics)
            
            logger.info(f"Ragas evaluation complete: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Ragas evaluation: {e}")
            raise
    
    def save_report(self, results: Dict, output_path: str):
        """
        Save evaluation report to file.
        
        Args:
            results: Evaluation results
            output_path: Path to save report
        """
        try:
            report = {
                'metrics': {
                    'faithfulness': float(results.get('faithfulness', 0)),
                    'answer_relevancy': float(results.get('answer_relevancy', 0)),
                    'context_recall': float(results.get('context_recall', 0)),
                    'context_precision': float(results.get('context_precision', 0))
                },
                'summary': {
                    'average_score': float(results.get('faithfulness', 0)),
                    'num_samples': len(results) if hasattr(results, '__len__') else 0
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise


def run_evaluation_from_yaml(yaml_path: str, rag_system, output_path: str = "reports/ragas_report.yaml"):
    """
    Run evaluation from YAML test dataset.
    
    Args:
        yaml_path: Path to test dataset YAML
        rag_system: RAG system instance with ask() method
        output_path: Path to save evaluation report
    """
    evaluator = RagasEvaluator()
    
    # Load test cases
    test_cases = evaluator.load_test_dataset(yaml_path)
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    # Run RAG system on each test case
    for test_case in test_cases:
        question = test_case['question']
        ground_truth = test_case.get('expected_answer', '')
        
        # Get answer from RAG system
        response = rag_system.ask(question)
        
        questions.append(question)
        answers.append(response['answer'])
        contexts.append([s['snippet'] for s in response['sources']])
        ground_truths.append(ground_truth)
    
    # Run evaluation
    results = evaluator.evaluate(
        questions,
        answers,
        contexts,
        ground_truths if ground_truths else None
    )
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_report(results, output_path)
    
    return results
