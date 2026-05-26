import os
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from judger import PreferenceCollector, PREFERENCE_ATTRIBUTES


class WinrateEvaluator:    
    def __init__(self, judge: PreferenceCollector, attribute: str):
        self.judge = judge
        self.attribute = attribute
        self.results = []
        
    def load_baseline_data(self, baseline_file: str) -> Dict[str, Dict]:
        baseline_data = {}
        
        if not os.path.exists(baseline_file):
            raise FileNotFoundError(f"Baseline file not found: {baseline_file}")
            
        with open(baseline_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    query = data.get('query', '').strip()
                    base_response = data.get('base', '').strip()
                    
                    if query and base_response:
                        baseline_data[query] = {
                            'index': data.get('index', line_num - 1),
                            'query': query,
                            'response': base_response
                        }
                except json.JSONDecodeError:
                    continue

        return baseline_data
        
    def load_ours_data(self, ours_file: str) -> List[Dict]:
        ours_data = []
        
        if not os.path.exists(ours_file):
            raise FileNotFoundError(f"Ours file not found: {ours_file}")
            
        with open(ours_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    query = data.get('query', '').strip()

                    ours_response = data.get('ours', '').strip()
                    if not ours_response:
                        ours_response = data.get('beam', '').strip()
                    if not ours_response:
                        ours_response = data.get('pref', '').strip()
                    if not ours_response:
                        ours_response = data.get('la', '').strip()

                    if query and ours_response:
                        ours_data.append({
                            'index': line_num - 1,
                            'query': query,
                            'response': ours_response
                        })
                except json.JSONDecodeError:
                    continue

        return ours_data
        
    def match_data(self, ours_data: List[Dict], baseline_data: Dict[str, Dict]) -> List[Tuple[Dict, Dict]]:
        matched_pairs = []
        unmatched_count = 0

        for ours_item in ours_data:
            query = ours_item['query']
            
            if query in baseline_data:
                baseline_item = baseline_data[query]
                matched_pairs.append((ours_item, baseline_item))
            else:
                unmatched_count += 1

        return matched_pairs
        
    def evaluate_pair(self, ours_item: Dict, baseline_item: Dict) -> Optional[Dict]:
        query = ours_item['query']
        ours_response = ours_item['response']
        baseline_response = baseline_item['response']
        
        example = {
            'attribute': self.attribute,
            'query': query,  # Original query for API prompt
            'enhanced_query': query,  # Same as query for consistency
            'response_1': ours_response,  # Our method response (A)
            'response_2': baseline_response  # Baseline response (B)
        }
        
        try:
            result = self.judge.collect_single(example)
            if result is None:
                return None
                
            ours_wins = (result['label'] == 0)
            
            return {
                'query': query,
                'ours_response': ours_response,
                'baseline_response': baseline_response,
                'ours_wins': ours_wins,
                'judgment_label': result['label']
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate pair: {e}")
            return None
            
    def evaluate_all(self, matched_pairs: List[Tuple[Dict, Dict]]) -> Dict:
        print(f"\nEvaluating {len(matched_pairs)} pairs using API judge...")
        print(f"Target attribute: {self.attribute}")
        print(f"Judge model: {self.judge.model}")
        
        wins = 0
        total = 0
        failed_evaluations = 0
        
        for ours_item, baseline_item in tqdm(matched_pairs, desc="Evaluating pairs"):
            result = self.evaluate_pair(ours_item, baseline_item)
            
            if result is None:
                failed_evaluations += 1
                continue
                
            self.results.append(result)
            total += 1
            
            if result['ours_wins']:
                wins += 1
                
            time.sleep(self.judge.sleep_time)
            
        winrate = wins / total if total > 0 else 0.0
        
        stats = {
            'wins': wins,
            'total': total,
            'winrate': winrate,
            'failed_evaluations': failed_evaluations,
            'attribute': self.attribute,
            'judge_model': self.judge.model
        }
        
        print(f"\nEvaluation Results:")
        print(f"Wins: {wins}/{total}")
        print(f"Winrate: {winrate:.3f}")
        print(f"Failed evaluations: {failed_evaluations}")
        
        return stats
        
    def save_results(self, output_file: str, stats: Dict, ours_file: str, baseline_file: str, model_name: str = None):
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        win_percentage = stats['winrate'] * 100
        loss_count = stats['total'] - stats['wins']
        loss_percentage = (loss_count / stats['total'] * 100) if stats['total'] > 0 else 0

        ours_basename = os.path.basename(ours_file)
        baseline_basename = os.path.basename(baseline_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Winrate Evaluation Results\n")
            f.write(f"========================\n")
            f.write(f"Evaluation Configuration:\n")
            f.write(f"  Target Attribute: {stats['attribute']}\n")
            f.write(f"  Judge Model: {stats['judge_model']}\n")
            if model_name:
                f.write(f"  Model Name: {model_name}\n")
            f.write(f"  Ours Dataset: {ours_basename}\n")
            f.write(f"  Baseline Dataset: {baseline_basename}\n")
            f.write(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n")
            f.write(f"Summary Statistics:\n")
            f.write(f"  Total Evaluations: {stats['total']}\n")
            f.write(f"  Ours Wins: {stats['wins']} ({win_percentage:.1f}%)\n")
            f.write(f"  Baseline Wins: {loss_count} ({loss_percentage:.1f}%)\n")
            f.write(f"  Winrate: {stats['winrate']:.3f}\n")
            f.write(f"  Failed Evaluations: {stats['failed_evaluations']}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate winrate between ours and baseline responses")
    
    parser.add_argument("--ours_file", required=True, 
                       help="Path to ours responses jsonl file")
    parser.add_argument("--baseline_file", required=True,
                       help="Path to baseline responses jsonl file")
    
    parser.add_argument("--attribute", required=True,
                       choices=list(PREFERENCE_ATTRIBUTES.keys()),
                       help="Target attribute for evaluation")
    
    parser.add_argument("--openai_api_key", required=True,
                       help="OpenAI API key for judge")
    parser.add_argument("--judge_model", required=True,
                       help="Judge model name")
    parser.add_argument("--api_base_url", required=True,
                       help="API base URL")
    parser.add_argument("--api_sleep_time", type=float, default=1.0,
                       help="Sleep time between API calls")
    parser.add_argument("--api_max_retries", type=int, default=3,
                       help="Maximum API retries")
    
    parser.add_argument("--output_file", 
                       help="Output file for results ")
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if not args.output_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ours_basename = os.path.splitext(os.path.basename(args.ours_file))[0]
        baseline_basename = os.path.splitext(os.path.basename(args.baseline_file))[0]
        args.output_file = f"winrate_{ours_basename}_vs_{baseline_basename}_{args.attribute}_{timestamp}.txt"
    
    print("Starting Winrate Evaluation")
    print("=" * 50)
    print(f"Ours file: {args.ours_file}")
    print(f"Baseline file: {args.baseline_file}")
    print(f"Attribute: {args.attribute}")
    print(f"Judge model: {args.judge_model}")
    print(f"Output file: {args.output_file}")
    print()
    
    try:
        # Initialize judge
        judge = PreferenceCollector(
            api_key=args.openai_api_key,
            model=args.judge_model,
            base_url=args.api_base_url,
            sleep_time=args.api_sleep_time,
            max_retries=args.api_max_retries
        )
        
        # Initialize evaluator
        evaluator = WinrateEvaluator(judge=judge, attribute=args.attribute)
        
        # Load data
        baseline_data = evaluator.load_baseline_data(args.baseline_file)
        ours_data = evaluator.load_ours_data(args.ours_file)
        
        # Match data
        matched_pairs = evaluator.match_data(ours_data, baseline_data)
        
        if len(matched_pairs) == 0:
            print("No matched pairs found. Check that queries match between files.")
            return
            
        # Evaluate
        stats = evaluator.evaluate_all(matched_pairs)
        
        # Save results
        evaluator.save_results(args.output_file, stats, args.ours_file, args.baseline_file)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Final winrate: {stats['winrate']:.3f} ({stats['wins']}/{stats['total']})")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
