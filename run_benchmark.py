import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.eval.pipeline import run_eval_pipeline

result = run_eval_pipeline(max_questions=None, save_results=True)
m = result['metrics']
g = result['gate_result']

print()
print('=== FINAL BENCHMARK RESULTS ===')
print(f"Total evaluated:    {m['total_evaluated']}")
print(f"Hallucination rate: {m['hallucination_rate']:.1%}  (target: <8%)")
print(f"Avg faithfulness:   {m['avg_faithfulness']:.2f}  (target: >0.88)")
print(f"Min faithfulness:   {m['min_faithfulness']:.2f}")
print(f"Hallucination count:{m['hallucination_count']}")
print(f"Deployment gate:    {'APPROVED' if g['approved'] else 'BLOCKED'}")