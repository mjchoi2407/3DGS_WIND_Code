"""성능, 독립 검산, 정확 일치 관측, 회귀 예산, 생산 적용을 분리한다."""

def judgement(baseline, candidate, audit, graph, comparisons):
    valid=bool(comparisons) and all(row.get('status')=='finite' and row.get('linf') is not None for row in comparisons)
    exact=all(row['linf']==0 for row in comparisons) if valid else None
    enough=bool(baseline and candidate and baseline['n']>=3 and candidate['n']>=3)
    improved=bool(enough and candidate['median']+candidate['mad']<baseline['median']-baseline['mad'])
    return dict(performance='improved' if improved else 'tie_or_inconclusive' if enough else 'not_measured',
        physical_audit='passed' if audit else 'failed_or_missing',
        exact_equality_observed=exact,comparison_status='complete' if valid else 'missing_or_invalid',
        numerical_regression='budget_not_defined' if valid else 'incomplete',budget_source=None,
        actual_graph='verified' if graph else 'not_verified',production_promotion='held',
        production_enabled=False,training_eligible=False)
