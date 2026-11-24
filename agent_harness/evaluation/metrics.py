import math


def ratio(numerator, denominator):
    return numerator/denominator if denominator else 0.0


def aggregate(records):
    total = len(records)
    successful = sum(r['success'] for r in records)
    verified = sum(r['verified'] for r in records)
    actions = sum(r['tool_calls'] for r in records)
    tokens = sum(r['tokens'] for r in records)
    return {
        'tasks':total, 'successful_tasks':successful, 'independently_verified_successes':verified,
        'success_rate':ratio(successful,total), 'verified_completion_rate':ratio(verified,total),
        'action_efficiency':ratio(successful,actions), 'token_efficiency':ratio(successful,tokens),
        'recovery_rate':ratio(sum(r['recovered'] for r in records),sum(r['recoverable_failures'] for r in records)),
        'human_intervention_rate':ratio(sum(r['human_intervention'] for r in records),total),
        'skill_reuse_rate':ratio(sum(r['verified'] and r['skill_reuse'] for r in records),sum(r['skill_eligible'] for r in records)),
        'cost_adjusted_performance':ratio(verified,tokens+100*actions),
        **{key:sum(r.get(key,0) for r in records) for key in (
            'latency','model_calls','recursive_calls','subagent_calls','tokens','estimated_cost',
            'memory_retrievals','memory_writes','accepted_changes','rejected_changes','rollback_count')},
        'failure_accumulation':total-verified,
    }
