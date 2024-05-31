yaml_code = [
    "services:",
    "  frontend:",
    "    image: gorisso/frontend_crypto:1.0.0",
    "    container_name: frontend",
    "    ports:",
    "      - '3000:3000'",
    "    environment:",
    "      NODE_ENV: production",
    "    restart: always",
    "  backend:",
    "    image: gorisso/backend_crypto:1.0.0",
    "    container_name: backend",
    "    ports:",
    "      - '7000:7000'",
    "    restart: always",
    "  bot:",
    "    image: gorisso/bot_crypto_crypto:1.0.0",
    "    container_name: bot",
    "    restart: always",
    "  redis:",
    "    image: redis",
    "    container_name: redis",
    "    ports:",
    "      - '6379:6379'",
    "    restart: always",
    "  nginx:",
    "    image: nginx",
    "    ports:",
    "      - '80:80'",
    "    volumes:",
    "      - ./nginx.conf:/etc/nginx/nginx.conf:ro",
    "      - /etc/nginx/certs/server.crt:/etc/nginx/certs/server.crt:ro",
    "      - /etc/nginx/certs/server.key:/etc/nginx/certs/server.key:ro",
    "    depends_on:",
    "      - frontend",
    "      - backend",
    "      - bot",
    "    restart: always"
]


yaml_code_combined = '\n'.join(yaml_code)
yaml = []
yaml.append(yaml_code_combined)

