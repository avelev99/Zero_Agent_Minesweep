from minesweeper_env import MinesweeperEnv, make_vec_env


def test_env_reset_hidden():
    env = MinesweeperEnv(size=(4, 4), n_mines=2, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (4, 4)
    assert (obs == -1).all()


def test_first_click_safe():
    env = MinesweeperEnv(size=(4, 4), n_mines=3, seed=0)
    env.reset()
    obs, reward, done, _, _ = env.step(0)
    assert env.board[0, 0] != 9
    assert obs[0, 0] >= 0
    assert not done


def test_flag_toggle():
    env = MinesweeperEnv(size=(4, 4), n_mines=1, seed=0)
    env.reset()
    flag_action = env.size[0] * env.size[1] + 0
    obs, _, _, _, _ = env.step(flag_action)
    assert obs[0, 0] == -2
    obs, _, _, _, _ = env.step(flag_action)
    assert obs[0, 0] == -1


def test_chord_action():
    env = MinesweeperEnv(size=(2, 2), n_mines=1, seed=0, first_click_safe=False)
    env.reset()
    env.step(0)  # reveal (0,0)
    flag_idx = env.size[0] * env.size[1] + 3  # flag mine at (1,1)
    env.step(flag_idx)
    chord_action = env.size[0] * env.size[1] * 2 + 0
    obs, reward, done, _, _ = env.step(chord_action)
    assert done and reward == 1.0
    assert (obs >= 0).sum() == 3  # three cells revealed


def test_vector_env():
    vec = make_vec_env(2, size=(4, 4), n_mines=1, seed=0)
    obs, _ = vec.reset()
    assert obs.shape == (2, 4, 4)


def test_action_mask():
    env = MinesweeperEnv(size=(2, 2), n_mines=0, seed=0)
    env.reset()
    mask = env.action_mask()
    cells = env.size[0] * env.size[1]
    assert mask[:cells].sum() == cells  # reveal valid everywhere
    assert mask[cells : 2 * cells].sum() == cells  # flag anywhere
    assert not mask[2 * cells :].any()  # no chord initially

    env.step(0)
    mask = env.action_mask()
    assert not mask[0]  # revealed cell can't be revealed again
    assert mask[2 * cells]  # chord now valid for revealed cell
