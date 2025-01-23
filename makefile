CHECKFILE = imports
SCRIPTFILE = script

all: help

check: ## Check system setup and issue message for missing python deps
	python $(CHECKFILE).py

run: ## Run main script to extract a 256x256 image
	python $(SCRIPTFILE).py

help: ## Show this help
	@printf "Usage:\033[36m make [target]\033[0m\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Phony targets:
.PHONY: check
.PHONY: help
