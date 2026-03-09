.PHONY: setup build run dev info docker docker-cuda clean

# ── Native (macOS Metal) ────────────────────────────────────
build:
	cargo build --release --features metal

run: build
	./target/release/fast-embed serve

dev:
	cargo run --features metal -- serve

info:
	cargo run --features metal -- info

# ── Docker (server deployment) ──────────────────────────────
docker:
	docker compose up --build -d

docker-cuda:
	docker compose -f docker-compose.cuda.yml up --build -d

# ── Setup & utilities ───────────────────────────────────────
setup:
	mkdir -p data models
	cp -n .env.example .env 2>/dev/null || true
	@echo "Done. Edit .env as needed, then run: make dev"

clean:
	cargo clean
