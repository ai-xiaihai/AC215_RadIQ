docker-compose up data-pipeline
docker-compose run -v "$(pwd):/src" data-pipeline bash

