.PHONY: build up down logs demo ps curl health ready

build:
	docker build -t edge-serve-ab:local .

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f app

ps:
	docker compose ps

curl:
	curl -s http://localhost:8080/healthz | jq .

health:
	curl -i http://localhost:8080/healthz

ready:
	curl -i http://localhost:8080/ready

demo: up
	@echo ""
	@echo "✅ Stack is coming up!"
	@echo "App:        http://localhost:8080"
	@echo "Healthz:    http://localhost:8080/healthz"
	@echo "Ready:      http://localhost:8080/ready"
	@echo "Metrics:    http://localhost:8080/metrics"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000 (admin/admin)"
	@echo ""
