#!/usr/bin/env bash
#
# deploy.sh -- Build and push the nixtla-sandbox Lambda image to ECR.
#
# Required environment variables (export before running, or set in CI):
#   AWS_ACCOUNT_ID     Your AWS account ID (e.g. 123456789012)
#   AWS_REGION         Target region (default: us-east-1)
#   LAMBDA_ROLE_ARN    IAM role ARN the Lambda will assume
#
# Optional:
#   IMAGE_TAG          Image tag (default: latest)
#
# Example:
#   export AWS_ACCOUNT_ID=123456789012
#   export AWS_REGION=us-east-1
#   export LAMBDA_ROLE_ARN=arn:aws:iam::123456789012:role/lambda-execution-role
#   ./deploy.sh
#
set -euo pipefail

: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is required (export it before running)}"
: "${LAMBDA_ROLE_ARN:?LAMBDA_ROLE_ARN is required (export it before running)}"

REGION="${AWS_REGION:-us-east-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ECR_REPO="nixtla-sandbox"
LAMBDA_FUNCTION="NixtlaSandbox"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

echo ">> Deploying ${LAMBDA_FUNCTION} (${ECR_URI})"

# 1. Login to AWS ECR
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# 2. Create ECR repository (no-op if it already exists)
aws ecr create-repository --repository-name "${ECR_REPO}" --region "${REGION}" >/dev/null 2>&1 || true

# 3. Build Docker image from the lambda/ directory
docker build -t "${ECR_REPO}:${IMAGE_TAG}" lambda/

# 4. Tag image for ECR
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}"

# 5. Push image to ECR
docker push "${ECR_URI}"

# 6. Create Lambda function (no-op if it already exists).
#    NOTE: this only creates the function. Updates to env vars, VPC config,
#    layers, etc. must be applied separately via `aws lambda update-function-configuration`.
aws lambda create-function \
  --function-name "${LAMBDA_FUNCTION}" \
  --package-type Image \
  --code "ImageUri=${ECR_URI}" \
  --role "${LAMBDA_ROLE_ARN}" \
  --timeout 60 \
  --memory-size 2048 \
  --region "${REGION}" >/dev/null 2>&1 || true

echo ">> Done. Image pushed: ${ECR_URI}"
