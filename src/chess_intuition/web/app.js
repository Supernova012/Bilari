const PIECES = {
  wk: "♔", wq: "♕", wr: "♖", wb: "♗", wn: "♘", wp: "♙",
  bk: "♚", bq: "♛", br: "♜", bb: "♝", bn: "♞", bp: "♟",
};
const PROMOTION_NAMES = { q: "Queen", r: "Rook", b: "Bishop", n: "Knight" };
const PROMOTION_GLYPHS = { q: "♛", r: "♜", b: "♝", n: "♞" };

const app = {
  config: null,
  active: "puzzle",
  puzzle: null,
  game: null,
  orientation: "w",
  selectedSquare: null,
  highlightedMove: null,
  busy: false,
  moveBusy: false,
  toastTimer: null,
  autoplayToken: 0,
  dragSource: null,
  suppressClick: false,
};

const $ = (id) => document.getElementById(id);

async function apiPost(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let payload;
  try {
    payload = await response.json();
  } catch (_) {
    throw new Error("The local app returned an unreadable response.");
  }
  if (!response.ok) throw new Error(payload.error || "The request could not be completed.");
  return payload;
}

function currentSession() {
  return app.active === "puzzle" ? app.puzzle : app.game;
}

function randomColor() {
  const bytes = new Uint8Array(1);
  crypto.getRandomValues(bytes);
  return bytes[0] & 1 ? "w" : "b";
}

function modeLabel(mode, seconds) {
  if (mode === "intuition") return "Intuition · no look-ahead";
  if (mode === "ultra") return `Ultra · up to ${seconds}s`;
  const budgets = { low: "0.25s", medium: "1s", high: "3s" };
  return `${mode[0].toUpperCase()}${mode.slice(1)} · ${budgets[mode]}`;
}

function modeSelectValue(id) {
  return $(id).value;
}

function ultraSeconds() {
  return Math.max(1, Math.min(30, Number.parseInt($("ultraSeconds").value, 10) || 10));
}

function sessionForPuzzle(state) {
  return {
    kind: "puzzle",
    state,
    initialFen: state.fen,
    moves: [],
    sanMoves: [],
    undoStack: [],
    solverColor: state.side_to_move,
    playStyle: $("puzzlePlayStyle").value,
  };
}

function sessionForGame(state, matchup, assignment, humanColor, modeByModel) {
  return {
    kind: "game",
    state,
    initialFen: state.fen,
    moves: [],
    sanMoves: [],
    undoStack: [],
    matchup,
    assignment,
    humanColor,
    modeByModel,
    paused: false,
  };
}

function sessionSnapshot(session) {
  session.undoStack.push({
    state: session.state,
    moves: session.moves.slice(),
    sanMoves: session.sanMoves.slice(),
  });
}

function sessionCanHumanMove(session) {
  if (!session?.state || session.state.game_over || app.moveBusy) return false;
  const sideToMove = session.state.side_to_move;
  if (session.kind === "puzzle") {
    return session.playStyle === "both" || sideToMove === session.solverColor;
  }
  // Pausing stops automatic engine replies; it should not prevent the player
  // from moving when the human is the side to move.
  return session.assignment?.[sideToMove] === "human";
}

function currentActor(session) {
  if (session?.kind === "puzzle") {
    if (session.playStyle === "both" || session.state.side_to_move === session.solverColor) return "human";
    return "bilari";
  }
  return session?.assignment?.[session.state.side_to_move] || "—";
}

function participantName(actor) {
  if (actor === "human") return "You";
  if (actor === "bilari") return "Bilari";
  if (actor === "oldb") return "Old-B";
  return "—";
}

function sideName(color) {
  return color === "w" ? "White" : "Black";
}

function setToast(message, success = false) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.toggle("success", success);
  toast.classList.remove("hidden");
  clearTimeout(app.toastTimer);
  app.toastTimer = setTimeout(() => toast.classList.add("hidden"), success ? 2600 : 4500);
}

function clearAnalysis() {
  app.highlightedMove = null;
  $("analysisTitle").textContent = "Candidate moves";
  $("analysisMeta").textContent = "Choose a position and ask for a suggestion.";
  $("analysisFooter").textContent = "";
  const list = $("analysisList");
  list.className = "analysis-list empty-analysis";
  list.innerHTML = '<div class="empty-glyph">♘</div><p>Your top five suggestions will appear here.</p>';
  renderBoard();
}

function showThinking(isThinking, text = "Thinking…") {
  app.busy = isThinking;
  $("spinner").classList.toggle("hidden", !isThinking);
  if (isThinking) $("analysisMeta").textContent = text;
  updateControls();
  renderBoard();
}

function updateAvailability() {
  const models = app.config?.models || {};
  for (const name of ["bilari", "oldb"]) {
    const info = models[name];
    const badge = $(`${name}Availability`);
    if (!badge) continue;
    badge.textContent = info?.available ? "Ready" : "Missing";
    badge.classList.toggle("unavailable", !info?.available);
  }
}

function updateControls() {
  const puzzleReady = Boolean(app.puzzle?.state);
  const puzzleEngineTurn = app.puzzle?.playStyle === "reply" && app.puzzle?.state.side_to_move !== app.puzzle?.solverColor;
  $("puzzleSuggest").disabled = !puzzleReady || app.busy || app.moveBusy || app.puzzle?.state.game_over || puzzleEngineTurn;
  $("loadPuzzle").disabled = app.busy || app.moveBusy;
  $("startGame").disabled = app.busy || app.moveBusy;
  $("copyFen").disabled = !currentSession()?.state;
  $("downloadPgn").disabled = !currentSession()?.state || app.moveBusy;
  $("undoPuzzle").disabled = app.active !== "puzzle" || !app.puzzle?.undoStack.length || app.busy;
  const gameReady = Boolean(app.game);
  $("startGame").classList.remove("hidden");
  $("startGame").textContent = gameReady ? "Start a new game" : "Start game";
  $("gameActions").classList.toggle("hidden", !gameReady);
  if (gameReady) {
    $("pauseGame").textContent = app.game.paused ? "Resume" : "Pause";
    $("undoGame").disabled = !app.game.undoStack.length || app.busy || app.moveBusy;
    $("suggestGame").disabled = app.busy || app.moveBusy || app.game.state.game_over;
    $("suggestGame").textContent = "Suggest move";
  }
  if (app.active === "game") $("undoPuzzle").disabled = true;
}

function updateModePanels() {
  const puzzleActive = app.active === "puzzle";
  $("puzzlePanel").classList.toggle("hidden", !puzzleActive);
  $("gamePanel").classList.toggle("hidden", puzzleActive);
  $("puzzleTab").classList.toggle("active", puzzleActive);
  $("gameTab").classList.toggle("active", !puzzleActive);
  $("puzzleTab").setAttribute("aria-selected", String(puzzleActive));
  $("gameTab").setAttribute("aria-selected", String(!puzzleActive));
  updateControls();
  renderSession();
}

function boardSquares(session) {
  if (!session?.state) return [];
  const bySquare = Object.fromEntries(session.state.pieces.map((piece) => [piece.square, piece]));
  const legalMoves = session.state.legal_moves || [];
  const files = app.orientation === "w" ? "abcdefgh".split("") : "hgfedcba".split("");
  const ranks = app.orientation === "w" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const squares = [];
  const suggested = app.highlightedMove ? [app.highlightedMove.slice(0, 2), app.highlightedMove.slice(2, 4)] : [];
  const last = session.state.last_move ? [session.state.last_move.slice(0, 2), session.state.last_move.slice(2, 4)] : [];
  const checkedKing = session.state.in_check
    ? session.state.pieces.find((piece) => piece.color === session.state.side_to_move && piece.kind === "k")?.square
    : null;
  const canMove = sessionCanHumanMove(session);

  ranks.forEach((rank, rowIndex) => {
    files.forEach((file, colIndex) => {
      const square = `${file}${rank}`;
      const piece = bySquare[square];
      const parity = ("abcdefgh".indexOf(file) + rank - 1) % 2;
      const button = document.createElement("button");
      button.type = "button";
      button.className = `square ${parity === 0 ? "dark" : "light"}`;
      button.dataset.square = square;
      button.setAttribute("aria-label", `${square}${piece ? ` ${piece.color === "w" ? "white" : "black"} ${piece.kind}` : " empty"}`);
      button.draggable = Boolean(canMove && piece && piece.color === session.state.side_to_move);
      if (!canMove || !piece || piece.color !== session.state.side_to_move) button.classList.add("readonly-square");
      if (app.selectedSquare === square) button.classList.add("selected");
      if (last.includes(square)) button.classList.add("last-move");
      if (suggested.includes(square)) button.classList.add("suggested-square");
      if (checkedKing === square) button.classList.add("checked-king");

      const isLegalTarget = app.selectedSquare && legalMoves.some((move) => move.uci.slice(0, 2) === app.selectedSquare && move.uci.slice(2, 4) === square);
      if (isLegalTarget) {
        const mark = document.createElement("span");
        mark.className = bySquare[square] ? "legal-capture" : "legal-dot";
        button.append(mark);
      }
      if (piece) {
        const glyph = document.createElement("span");
        glyph.className = `piece ${piece.color === "w" ? "white-piece" : "black-piece"}`;
        glyph.textContent = PIECES[piece.color + piece.kind];
        button.append(glyph);
      }
      const showRank = colIndex === 0;
      const showFile = rowIndex === 7;
      if (showRank) {
        const coord = document.createElement("span");
        coord.className = "coord";
        coord.textContent = String(rank);
        button.append(coord);
      }
      if (showFile) {
        const coord = document.createElement("span");
        coord.className = "coord file-coord";
        coord.textContent = file;
        button.append(coord);
      }
      button.addEventListener("click", () => onSquareClick(square));
      button.addEventListener("dragstart", (event) => {
        if (!button.draggable) {
          event.preventDefault();
          return;
        }
        app.dragSource = square;
        app.selectedSquare = square;
        button.classList.add("dragging");
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", square);
      });
      button.addEventListener("dragover", (event) => {
        if (app.dragSource) event.preventDefault();
      });
      button.addEventListener("drop", (event) => {
        event.preventDefault();
        const source = app.dragSource || event.dataTransfer.getData("text/plain");
        app.dragSource = null;
        app.suppressClick = true;
        setTimeout(() => { app.suppressClick = false; }, 300);
        if (source) moveFromTo(source, square);
      });
      button.addEventListener("dragend", () => {
        button.classList.remove("dragging");
        if (app.dragSource === square) {
          app.dragSource = null;
          app.selectedSquare = null;
          renderBoard();
        }
      });
      squares.push(button);
    });
  });
  return squares;
}

function renderBoard() {
  const board = $("board");
  board.replaceChildren(...boardSquares(currentSession()));
  const session = currentSession();
  if (!session?.state) {
    renderPlayerStrips(null);
    $("turnLabel").textContent = "Load a position to begin";
    $("turnDetail").textContent = "The side to move is read from the FEN.";
    $("boardStatus").textContent = "Drag a piece, or click it and then a highlighted square.";
    $("currentFen").value = "";
    return;
  }

  $("currentFen").value = session.state.fen;
  renderPlayerStrips(session);
  const side = sideName(session.state.side_to_move);
  if (session.state.game_over) {
    $("turnLabel").textContent = session.state.result === "1/2-1/2" ? "Draw" : `${sideName(session.state.winner)} wins`;
    $("turnDetail").textContent = session.state.termination || "Game over";
    $("boardStatus").textContent = `Result ${session.state.result}`;
  } else {
    $("turnLabel").textContent = `${side} to move${session.state.in_check ? " · check" : ""}`;
    if (app.active === "game") {
      const actor = currentActor(session);
      $("turnDetail").textContent = session.paused ? `Paused · ${participantName(actor)} to move` : `${participantName(actor)} to move as ${side.toLowerCase()}`;
    } else if (session.playStyle === "reply" && session.state.side_to_move !== session.solverColor) {
      $("turnDetail").textContent = "Bilari is replying for the other side.";
    } else {
      $("turnDetail").textContent = session.playStyle === "both" ? "You can move either color." : "Your turn to solve.";
    }
    $("boardStatus").textContent = app.busy ? `${participantName(currentActor(session))} is thinking…` : "Drag a piece, or click it and then a highlighted square.";
  }
  if (session.state.warning) $("boardStatus").textContent = session.state.warning;
  renderMoveHistory(session);
  renderGameResult(session);
  updateControls();
}

function playerName(session, color) {
  if (session.kind === "puzzle") {
    if (session.playStyle === "both" || color === session.solverColor) return "You";
    return "Bilari";
  }
  return participantName(session.assignment[color]);
}

function renderPlayerStrips(session) {
  for (const [id, color] of [["topPlayer", app.orientation === "w" ? "b" : "w"], ["bottomPlayer", app.orientation]]) {
    const strip = $(id);
    strip.replaceChildren();
    if (!session?.state) {
      strip.classList.add("hidden");
      continue;
    }
    strip.classList.remove("hidden");
    strip.classList.toggle("active-player", session.state.side_to_move === color);
    const avatar = document.createElement("span");
    avatar.className = `player-avatar ${color === "w" ? "white-avatar" : "black-avatar"}`;
    avatar.textContent = color === "w" ? "♔" : "♚";
    const info = document.createElement("span");
    const name = document.createElement("span");
    name.className = "player-name";
    name.textContent = playerName(session, color);
    const side = document.createElement("span");
    side.className = "player-side";
    side.textContent = `${sideName(color)}${session.state.side_to_move === color ? " · to move" : ""}`;
    info.append(name, side);
    strip.append(avatar, info);
    if (session.state.side_to_move === color && !session.state.game_over) {
      const turn = document.createElement("span");
      turn.className = "to-move-chip";
      turn.textContent = "TURN";
      strip.append(turn);
    }
  }
}

function renderSession() {
  renderBoard();
  updateControls();
}

function renderMoveHistory(session) {
  const history = $("moveHistory");
  history.replaceChildren();
  if (!session.sanMoves.length) {
    const empty = document.createElement("span");
    empty.className = "history-empty";
    empty.textContent = "Moves will appear here.";
    history.append(empty);
    return;
  }
  session.sanMoves.forEach((move) => {
    const row = document.createElement("div");
    row.className = "history-row";
    const number = document.createElement("span");
    number.className = "history-number";
    number.textContent = move.side === "w" ? `${move.number}.` : `${move.number}…`;
    const san = document.createElement("span");
    san.className = "history-san";
    san.textContent = move.san;
    row.append(number, san);
    history.append(row);
  });
  history.scrollTop = history.scrollHeight;
}

function renderGameResult(session) {
  const box = $("gameResult");
  if (!session.state.game_over) {
    box.classList.add("hidden");
    box.textContent = "";
    return;
  }
  box.classList.remove("hidden");
  const winner = session.state.winner ? `${sideName(session.state.winner)} wins` : "Draw";
  box.textContent = `${winner} · ${session.state.result} · ${session.state.termination || "Game over"}`;
}

async function loadPuzzle() {
  try {
    const fen = $("puzzleFen").value.trim();
    const state = await apiPost("/api/state", { fen });
    app.autoplayToken += 1;
    app.puzzle = sessionForPuzzle(state);
    app.orientation = state.side_to_move;
    app.selectedSquare = null;
    clearAnalysis();
    renderSession();
    setToast(`${sideName(state.side_to_move)} to move. Position loaded.`, true);
  } catch (error) {
    setToast(error.message);
  }
}

function updateMatchupFields() {
  const matchup = $("matchup").value;
  const needsOldB = matchup === "bilari-oldb" || matchup === "me-oldb";
  $("oldbModeWrap").classList.toggle("hidden", !needsOldB);
  $("oldbNote").classList.toggle("hidden", !needsOldB);
  $("bilariModeWrap").classList.toggle("hidden", matchup === "me-oldb");
}

async function startGame() {
  try {
    let fen = $("gameFen").value.trim();
    if (!fen) fen = app.config.starting_fen;
    const state = await apiPost("/api/state", { fen });
    const matchup = $("matchup").value;
    const assignment = {};
    let humanColor = null;
    if (matchup === "bilari-oldb") {
      const bilariColor = randomColor();
      assignment[bilariColor] = "bilari";
      assignment[bilariColor === "w" ? "b" : "w"] = "oldb";
    } else {
      humanColor = randomColor();
      assignment[humanColor] = "human";
      assignment[humanColor === "w" ? "b" : "w"] = matchup === "me-bilari" ? "bilari" : "oldb";
    }
    const modeByModel = {
      bilari: $("bilariMode").value,
      oldb: $("oldbMode").value,
    };
    app.autoplayToken += 1;
    app.game = sessionForGame(state, matchup, assignment, humanColor, modeByModel);
    app.orientation = humanColor || (assignment.w === "bilari" ? "w" : "b");
    app.selectedSquare = null;
    app.active = "game";
    updateModePanels();
    clearAnalysis();
    const colorLine = Object.entries(assignment)
      .map(([color, actor]) => `${participantName(actor)} ${sideName(color)}`)
      .join(" · ");
    setToast(`Sides assigned randomly: ${colorLine}.`, true);
    if (matchup === "bilari-oldb") void runGameLoop();
    else if (currentActor(app.game) !== "human") void runGameLoop();
  } catch (error) {
    setToast(error.message);
  }
}

function gameModeForModel(session, model) {
  return session.modeByModel[model] || "high";
}

function modelForSuggestion(session) {
  if (session.kind === "puzzle") return "bilari";
  const actor = currentActor(session);
  if (actor === "bilari" || actor === "oldb") return actor;
  return session.matchup === "me-oldb" ? "oldb" : "bilari";
}

function modeForSuggestion(session, model) {
  return session.kind === "puzzle" ? $("puzzleMode").value : gameModeForModel(session, model);
}

async function analyzePosition(session, model, mode) {
  return apiPost("/api/analyze", {
    fen: session.state.fen,
    initial_fen: session.initialFen,
    moves: session.moves,
    model,
    mode,
    ultra_max_seconds: ultraSeconds(),
    top: 5,
  });
}

async function suggestMove() {
  const session = currentSession();
  if (!session?.state || session.state.game_over) return;
  if (session.kind === "game" && currentActor(session) !== "human" && !session.paused) {
    setToast("Pause model-vs-model play before asking for its current move.");
    return;
  }
  const model = modelForSuggestion(session);
  const mode = modeForSuggestion(session, model);
  const token = app.autoplayToken;
  const analyzedFen = session.state.fen;
  try {
    showThinking(true, `${participantName(model)} · ${modeLabel(mode, ultraSeconds())} · ranking moves…`);
    const result = await analyzePosition(session, model, mode);
    if (currentSession() !== session || token !== app.autoplayToken || session.state.fen !== analyzedFen) return;
    showAnalysis(result);
  } catch (error) {
    setToast(error.message);
  } finally {
    if (token === app.autoplayToken) showThinking(false);
  }
}

function showAnalysis(result) {
  app.highlightedMove = result.moves?.[0]?.uci || null;
  $("analysisTitle").textContent = `${result.model_label} suggestions`;
  $("analysisMeta").textContent = `${result.side_to_move} to move · ${result.mode_label} · ${result.thinking_time_ms.toFixed(0)} ms`;
  const list = $("analysisList");
  list.className = "analysis-list";
  list.replaceChildren();
  if (!result.moves?.length) {
    list.classList.add("empty-analysis");
    list.textContent = "No legal moves are available.";
  } else {
    result.moves.forEach((move) => {
      const row = document.createElement("div");
      row.className = "analysis-row";
      const pick = document.createElement("button");
      pick.type = "button";
      pick.className = "move-pick";
      const rank = document.createElement("span");
      rank.className = "rank-chip";
      rank.textContent = String(move.rank);
      const san = document.createElement("span");
      san.className = "move-san";
      san.textContent = move.san;
      const extra = document.createElement("span");
      extra.className = "move-extra";
      const detail = [];
      if (move.score_cp !== null && move.score_cp !== undefined) {
        if (Math.abs(move.score_cp) >= 99000) detail.push(move.score_cp > 0 ? "Mate" : "Mated");
        else detail.push(`${move.score_cp >= 0 ? "+" : ""}${(move.score_cp / 100).toFixed(1)}`);
      }
      if (move.policy_probability !== null && move.policy_probability !== undefined) {
        detail.push(`${(move.policy_probability * 100).toFixed(1)}% policy`);
      }
      extra.textContent = detail.join(" · ");
      pick.append(rank, san, extra);
      pick.addEventListener("click", () => {
        app.highlightedMove = move.uci;
        renderBoard();
      });
      const play = document.createElement("button");
      play.type = "button";
      play.className = "play-suggestion";
      play.textContent = "Play";
      play.addEventListener("click", async () => {
        if (!sessionCanHumanMove(session)) {
          if (session.kind === "game" && session.paused && currentActor(session) !== "human") {
            await applyMove(session, move.uci);
            return;
          }
          setToast("This side is controlled by the engine right now.");
          return;
        }
        await applyMove(session, move.uci);
      });
      row.append(pick, play);
      list.append(row);
    });
  }
  const parts = [];
  if (result.completed_depth) parts.push(`Depth ${result.completed_depth}`);
  if (result.nodes_searched) parts.push(`${result.nodes_searched.toLocaleString()} nodes`);
  if (result.evaluator) parts.push(result.evaluator);
  if (result.mode === "intuition") parts.push("0 search calls");
  if (result.timed_out) parts.push("time limit reached");
  $("analysisFooter").textContent = parts.join(" · ");
  renderBoard();
}

async function applyMove(session, uci, fromEngine = false) {
  if (!session?.state || session.state.game_over || app.moveBusy) return;
  const legal = session.state.legal_moves.find((item) => item.uci === uci);
  if (!legal) {
    setToast("That suggested move is no longer legal in this position.");
    return;
  }
  app.moveBusy = true;
  updateControls();
  try {
    const previous = session.state;
    const mover = fromEngine
      ? session.kind === "puzzle"
        ? "Bilari"
        : participantName(session.assignment[previous.side_to_move])
      : "You";
    const state = await apiPost("/api/move", {
      fen: previous.fen,
      initial_fen: session.initialFen,
      moves: session.moves,
      uci,
    });
    if (currentSession() !== session) return;
    sessionSnapshot(session);
    session.moves.push(uci);
    session.sanMoves.push({ san: legal.san, side: previous.side_to_move, number: Number(previous.fen.split(" ")[5]) || 1 });
    session.state = state;
    app.selectedSquare = null;
    app.highlightedMove = null;
    if (!fromEngine) {
      app.autoplayToken += 1;
      app.busy = false;
      $("spinner").classList.add("hidden");
    }
    clearAnalysis();
    renderSession();
    if (state.warning) setToast(state.warning);
    else if (fromEngine) $("boardStatus").textContent = `${mover} played ${legal.san}.`;
    if (!fromEngine && session.kind === "puzzle" && session.playStyle === "reply") void runPuzzleReply();
    if (!fromEngine && session.kind === "game") void runGameLoop();
  } catch (error) {
    setToast(error.message);
  } finally {
    app.moveBusy = false;
    updateControls();
    renderBoard();
  }
}

async function runGameLoop() {
  const session = app.game;
  if (!session || session.state.game_over || session.paused) return;
  const token = ++app.autoplayToken;
  while (token === app.autoplayToken && app.game === session && !session.state.game_over && !session.paused) {
    const actor = currentActor(session);
    if (actor === "human") break;
    const mode = gameModeForModel(session, actor);
    try {
      showThinking(true, `${participantName(actor)} · ${modeLabel(mode, ultraSeconds())} · thinking…`);
      const result = await analyzePosition(session, actor, mode);
      if (token !== app.autoplayToken || app.game !== session || session.paused) break;
      if (!result.moves?.length) break;
      showAnalysis(result);
      await applyMove(session, result.moves[0].uci, true);
      await new Promise((resolve) => setTimeout(resolve, 250));
    } catch (error) {
      setToast(error.message);
      session.paused = true;
      break;
    } finally {
      if (token === app.autoplayToken) showThinking(false);
    }
  }
  renderBoard();
}

async function runPuzzleReply() {
  const session = app.puzzle;
  if (!session || session.playStyle !== "reply" || session.state.game_over || session.state.side_to_move === session.solverColor) return;
  const token = ++app.autoplayToken;
  try {
    const mode = $("puzzleMode").value;
    showThinking(true, `Bilari · ${modeLabel(mode, ultraSeconds())} · replying…`);
    const result = await analyzePosition(session, "bilari", mode);
    if (token !== app.autoplayToken || app.puzzle !== session) return;
    showAnalysis(result);
    if (result.moves?.length) await applyMove(session, result.moves[0].uci, true);
  } catch (error) {
    setToast(error.message);
  } finally {
    if (token === app.autoplayToken) showThinking(false);
  }
}

function squaresByClick(square) {
  const session = currentSession();
  if (!session?.state || !sessionCanHumanMove(session)) {
    if (session?.state && !session.state.game_over) setToast("Wait for your turn, or pause model-vs-model play.");
    return;
  }
  if (app.selectedSquare && app.selectedSquare !== square && moveFromTo(app.selectedSquare, square)) {
    return;
  }
  const piece = session.state.pieces.find((item) => item.square === square);
  if (piece && piece.color === session.state.side_to_move) {
    app.selectedSquare = square;
    app.highlightedMove = null;
  } else {
    app.selectedSquare = null;
  }
  renderBoard();
}

function moveFromTo(from, to) {
  const session = currentSession();
  if (!session?.state || !sessionCanHumanMove(session)) return false;
  const matches = session.state.legal_moves.filter((move) => move.uci.slice(0, 2) === from && move.uci.slice(2, 4) === to);
  if (!matches.length) return false;
  if (matches.length === 1) void applyMove(session, matches[0].uci);
  else showPromotionChoice(matches);
  return true;
}

function showPromotionChoice(moves) {
  const box = $("promotionChoices");
  box.replaceChildren();
  const side = currentSession().state.side_to_move;
  moves.forEach((move) => {
    const suffix = move.uci.at(-1);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "promotion-choice";
    button.title = PROMOTION_NAMES[suffix];
    const colorGlyph = PROMOTION_GLYPHS[suffix];
    button.textContent = side === "w" ? colorGlyph.replace(/[♛♜♝♞]/, (symbol) => ({ "♛": "♕", "♜": "♖", "♝": "♗", "♞": "♘" }[symbol])) : colorGlyph;
    button.addEventListener("click", () => {
      $("promotionBackdrop").classList.add("hidden");
      void applyMove(currentSession(), move.uci);
    });
    box.append(button);
  });
  $("promotionBackdrop").classList.remove("hidden");
}

function onSquareClick(square) {
  if (app.suppressClick) return;
  squaresByClick(square);
}

function undoSession(session, plies = 1) {
  if (!session?.undoStack.length || app.busy || app.moveBusy) return;
  let previous = null;
  for (let i = 0; i < plies && session.undoStack.length; i += 1) previous = session.undoStack.pop();
  session.state = previous.state;
  session.moves = previous.moves;
  session.sanMoves = previous.sanMoves;
  app.selectedSquare = null;
  app.highlightedMove = null;
  clearAnalysis();
  renderSession();
}

function pauseGame() {
  if (!app.game) return;
  app.game.paused = !app.game.paused;
  app.autoplayToken += 1;
  app.busy = false;
  $("spinner").classList.add("hidden");
  updateControls();
  renderBoard();
  if (!app.game.paused) void runGameLoop();
}

function switchTab(mode) {
  if (mode !== app.active) {
    app.autoplayToken += 1;
    app.busy = false;
    $("spinner").classList.add("hidden");
  }
  app.active = mode;
  app.selectedSquare = null;
  clearAnalysis();
  updateModePanels();
  if (mode === "game" && app.game && !app.game.paused && currentActor(app.game) !== "human") void runGameLoop();
}

async function copyFen() {
  const session = currentSession();
  if (!session?.state) return;
  try {
    await navigator.clipboard.writeText(session.state.fen);
    setToast("Current FEN copied.", true);
  } catch (_) {
    $("currentFen").select();
    document.execCommand("copy");
    setToast("Current FEN copied.", true);
  }
}

async function downloadPgn() {
  const session = currentSession();
  if (!session?.state) return;
  try {
    const result = await apiPost("/api/pgn", {
      fen: session.state.fen,
      initial_fen: session.initialFen,
      moves: session.moves,
      event: app.active === "puzzle" ? "Chess Study Puzzle" : "Chess Study Game",
      white: playerName(session, "w"),
      black: playerName(session, "b"),
    });
    const blob = new Blob([result.pgn], { type: "application/x-chess-pgn;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `Chess-Study-${new Date().toISOString().replace(/[:.]/g, "-")}.pgn`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
    setToast("PGN created and downloaded.", true);
  } catch (error) {
    setToast(error.message);
  }
}

function initializeEvents() {
  $("puzzleTab").addEventListener("click", () => switchTab("puzzle"));
  $("gameTab").addEventListener("click", () => switchTab("game"));
  $("loadPuzzle").addEventListener("click", () => void loadPuzzle());
  $("puzzleSuggest").addEventListener("click", () => void suggestMove());
  $("startGame").addEventListener("click", () => void startGame());
  $("matchup").addEventListener("change", updateMatchupFields);
  $("suggestGame").addEventListener("click", () => void suggestMove());
  $("pauseGame").addEventListener("click", pauseGame);
  $("undoPuzzle").addEventListener("click", () => undoSession(app.puzzle, app.puzzle?.playStyle === "reply" ? 2 : 1));
  $("undoGame").addEventListener("click", () => {
    if (app.game && !app.game.paused) pauseGame();
    undoSession(app.game, 2);
  });
  $("rotateBoard").addEventListener("click", () => {
    app.orientation = app.orientation === "w" ? "b" : "w";
    renderBoard();
  });
  $("copyFen").addEventListener("click", () => void copyFen());
  $("downloadPgn").addEventListener("click", () => void downloadPgn());
  $("ultraSeconds").addEventListener("input", () => {
    $("ultraValue").textContent = String(ultraSeconds());
  });
  $("puzzlePlayStyle").addEventListener("change", () => {
    if (app.puzzle) {
      app.puzzle.playStyle = $("puzzlePlayStyle").value;
      app.autoplayToken += 1;
      renderBoard();
      if (app.puzzle.playStyle === "reply") void runPuzzleReply();
    }
  });
  $("cancelPromotion").addEventListener("click", () => $("promotionBackdrop").classList.add("hidden"));
}

async function initialize() {
  initializeEvents();
  updateMatchupFields();
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    if (!response.ok) throw new Error("Could not connect to the local chess app.");
    app.config = await response.json();
    updateAvailability();
    $("puzzleFen").value = app.config.starting_fen;
    const state = await apiPost("/api/state", { fen: app.config.starting_fen });
    app.puzzle = sessionForPuzzle(state);
    app.orientation = state.side_to_move;
    renderSession();
  } catch (error) {
    setToast(error.message);
    $("turnLabel").textContent = "The local app is not connected";
    $("turnDetail").textContent = "Close and reopen the Chess Study launcher.";
  }
}

initialize();
