

// // represents a state of a link
// class GameState {
// public:
//   virtual ~GameState() = default;

//   // [PHANTOM API] - Ask the engine for valid moves
//   virtual std::vector<Action> get_legal_actions() const = 0;

//   // [PHANTOM API] - Ask the engine to apply a move and return a NEW state
//   virtual std::unique_ptr<GameState>
//   apply_action(const Action &action) const = 0;

//   // [PHANTOM API] - terminal condition (is this real even now)?
//   virtual bool is_terminal() const = 0;

//   // debug printing
//   // virtual void print() const = 0;
// };

// // output of alphaknot
// struct ModelOutput {
//   double value; // [-1, 1] (1 = Unknot found, -1 = Dead end)
//   std::vector<double> policy_logits; // Raw scores for each action
// };

// // alphaknot
// namespace AlphaKnotAPI {
// // [PHANTOM API] - Call your PyTorch/TensorFlow model here
// ModelOutput evaluate(const GameState &state,
//                      const std::vector<Action> &legal_moves) {

//   // Mock return for compilation:
//   ModelOutput output;
//   output.value = -0.1; // Slight negative bias for non-terminal
//   output.policy_logits.resize(legal_moves.size(), 1.0); // Uniform distribution
//   return output;
// }
// } // namespace AlphaKnotAPI

// // ==================================================================================
// // actual MCTS code starts here
// // ==================================================================================

// struct MCTSNode {

//   double value_sum = 0.0;
//   double prior = 0.0; // p(s,a)

//   int32_t parent = -1;
//   int32_t first_child = -1;  // Head of linked list of children
//   int32_t next_sibling = -1; // Next node in the sibling list

//   // Action (4 bytes + padding typically)
//   Action action_leading_here;

//   // Metadata (4 bytes)
//   int32_t visits = 0;

//   // Flags (1 byte)
//   bool is_expanded = false;

//   // Constructor
//   MCTSNode(int32_t p, Action a, double prior_p)
//       : value_sum(0.0), prior(prior_p), parent(p), first_child(-1),
//         next_sibling(-1), action_leading_here(a), visits(0),
//         is_expanded(false) {}

//   // U-Value: The exploration bonus (PUCT formula)
//   double u_value(double parent_sqrt_visits, double cpuct) const {
//     return cpuct * prior * parent_sqrt_visits / (1.0 + visits);
//   }
// };