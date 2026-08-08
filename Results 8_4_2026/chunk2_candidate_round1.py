class GameModel:
    """Safe no-op stub substituted in when this round's LLM output could not
    be validated as safe code (SyntaxError or a disallowed import) -- see
    _validate_candidate_code. Always predicts no change and no goal, so it
    scores low but never crashes the sandboxed backtest."""
    def predict(self, grid_before, action, previous_state):
        return grid_before, False, previous_state
