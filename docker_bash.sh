# Insufficient number of arguments
if [ $# -lt 1 ]; then
    echo "Usage: ./run_docker.sh [run|exec|build|stop|remove]"
    exit 1
fi

case $1 in
    run)
        # Run the docker container
        docker run -v ./:/app/ --rm --gpus device=$CUDA_VISIBLE_DEVICES -d -it --name folbenchmark-container folbenchmark
        ;;
    exec)
        # Execute the models inside the docker container
        docker exec -it folbenchmark-container bash      
        ;;
    build)
        # Build the docker
        docker build ./ -t folbenchmark
        ;;
    stop)
        # Stop the docker container
        docker stop folbenchmark-container
        ;;
    remove)
        # Remove the docker container
        docker stop folbenchmark-container &&
        docker remove folbenchmark-container
        ;;
    *)
        # Invalid argument
        echo "Invalid argument"
        ;;
esac