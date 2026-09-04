FROM public.ecr.aws/lambda/python:3.11

# Install pre-compiled wheels (no GCC/yum needed)
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy Lambda handler
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}

# Handler entrypoint
CMD ["lambda_handler.lambda_handler"]
