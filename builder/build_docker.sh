#!/bin/bash

token=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:marciejj/plastic_detection_model:pull" | jq -r .token)
tags=$(curl -s -H "Authorization: Bearer $token" https://registry-1.docker.io/v2/marciejj/plastic_detection_model/tags/list | jq -r .tags[])

latest_tag=$(echo "$tags" | sort -V | tail -n 1)

IFS='.' read -ra parts <<< "$latest_tag"
parts[2]=$((parts[2] + 1))

new_version="${parts[0]}.${parts[1]}.${parts[2]}"

echo "New version is: $new_version"

docker build -t "marciejj/plastic_detection_model:$new_version" .

docker push "marciejj/plastic_detection_model:$new_version"
