sudo swapoff -a
sudo sed -ri 's/^[^#]*swap/#&/' /etc/fstab

# kernel modules for containers + kube networking
cat << 'EOF' | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF
sudo modprobe overlay
sudo modprobe br_netfilter

# sysctls
cat << 'EOF' | sudo tee /etc/sysctl.d/99-kubernetes-cri.conf
net.bridge.bridge-nf-call-iptables=1
net.bridge.bridge-nf-call-ip6tables=1
net.ipv4.ip_forward=1
EOF
sudo sysctl --system

sudo systemctl stop firewalld
sudo systemctl disable firewalld

# bash completion (helpful!)
sudo apt-get install -y bash-completion
echo 'source <(kubectl completion bash)' >>~/.bashrc
source ~/.bashrc

# manager and token
curl -sfL https://get.k3s.io | sudo sh -
sudo cat /var/lib/rancher/k3s/server/node-token

sudo chmod 777 /etc/rancher/k3s/k3s.yaml

mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config
chmod 600 ~/.kube/config

export KUBECONFIG=~/.kube/config

sudo apt update
sudo apt install -y bash-completion net-tools

kubectl get pods -A

# workers:
curl -sfL https://get.k3s.io | \
  sudo sh -s - agent \
    --server https://10.52.2.108:6443 \
    --token "K106c047a20d7c2585c93c2d6a3be1d27e5752c7b66af043d533303ef837a3ec876::server:01531294a3db10435cbfd8cb0bdfa7fb"

sudo chmod 777 /etc/rancher/k3s/k3s.yaml
kubectl get nodes

# k3s cider setup
sudo mkdir -p /etc/rancher/k3s
sudo vim /etc/rancher/k3s/config.yaml
sudo systemctl restart k3s
sudo systemctl restart k3s-agent

# content
kubelet-arg:
  - "max-pods=300"
  - "cpu-manager-policy=static"
  - "cpu-manager-policy-options=full-pcpus-only=true"
  - "topology-manager-policy=restricted"
  - "reserved-cpus=0,80,1,81"

kubelet-arg:
  - "max-pods=300"

cluster-cidr: 10.42.0.0/16
service-cidr: 10.43.0.0/16

kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"  "}{.spec.podCIDR}{"\n"}{end}'

kubectl describe node | grep -A5 -i pods
kubectl get pods -A -o wide | head -50

# taints
kubectl taint node intel-manager node-role.kubernetes.io/control-plane=true:NoSchedule
kubectl describe node intel-manager | grep -i taints -A3

# helm
curl -sSLf https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash

# fission
export FISSION_NAMESPACE="fission"
kubectl create namespace $FISSION_NAMESPACE
kubectl create -k "github.com/fission/fission/crds/v1?ref=v1.22.0"
helm repo add fission-charts https://fission.github.io/fission-charts/
helm repo update
helm install --version 1.22.1 --namespace $FISSION_NAMESPACE fission \
  --set serviceType=NodePort,routerServiceType=NodePort,logger.enableSecurityContext=true \
  fission-charts/fission-all

helm upgrade fission fission-charts/fission-all --namespace fission -f prometheus.yaml

## prothemus
export METRICS_NAMESPACE="monitoring"
kubectl create namespace $METRICS_NAMESPACE
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring
helm upgrade fission fission-charts/fission-all --namespace fission -f fission-prometheus.yaml
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring \
  -f prometheus.yaml
kubectl -n monitoring patch svc prometheus-kube-prometheus-prometheus \
  --type='merge' \
  -p '{
    "spec": {
      "type": "NodePort",
      "ports": [
        { "name": "http-web", "port": 9090, "targetPort": 9090, "nodePort": 30990, "protocol": "TCP" },
        { "name": "http-reloader", "port": 8080, "targetPort": 8080, "protocol": "TCP" }
      ]
    }
  }'


# functions
fission env create --name python --version 3 --poolsize 1 \
  --image ghcr.io/fission/python-env --mincpu 100 --maxcpu 200 \
  --minmemory 128 --maxmemory 256 --builder ghcr.io/fission/python-builder

fission pkg create --sourcearchive demo-src-pkg.zip \
  --env python --buildcmd "./build.sh" --name demo

fission pkg info --name ml-image-processing

fission fn create --name hello \
  --env python --src demo-src-pkg.zip  --entrypoint "hello.main" --buildcmd "./build.sh"

fission fn create --name hello --pkg demo --entrypoint "hello.main" \
  --env python --con 5 --idletimeout 5 --rpp 10

fission fn create --name ml-image-processing --pkg ml-image-processing --entrypoint "ml_image_processing.main" --env python \
  --mincpu 200 --maxcpu 500 --minmemory 256 --maxmemory 512 \
  --con 10

fission fn create --name hello --pkg demo --entrypoint "hello.main" --env python \
  --executortype newdeploy \--minscale 1 --maxscale 5 --mincpu 1000 \
  --maxcpu 2000 --minmemory 256 --maxmemory 512

fission route create --name hello-route \
  --function hello --url /hello --method POST


# custom runtime image
docker build -f Dockerfile-buster -t kevinjieuw114514/python-env-slim --build-arg PY_BASE_IMG=python:3.11-slim .
docker push kevinjieuw114514/python-env-slim

# ml-image-processing runtime image 
docker build -f Dockerfile-buster -t kevinjieuw114514/python-ml-image-processing --build-arg PY_BASE_IMG=python:3.11-slim .
docker push kevinjieuw114514/python-ml-image-processing

# builder
docker build -f Dockerfile-debian -t kevinjieuw114514/python-builder-slim .
docker push kevinjieuw114514/python-builder-slim


# v3 custom env
fission environment create --name pytorch --image kevinjieuw114514/python-env-slim \
  --builder kevinjieuw114514/python-builder-slim  --version 3 --poolsize 1
# ml-detection pipeline yolo env
fission environment create --name yolo --image kevinjieuw114514/python-torch-yolo \
  --builder kevinjieuw114514/python-builder-slim  --version 3 --poolsize 1
# ml-image-processing env
fission environment create --name ml-image-processing --image kevinjieuw114514/python-ml-image-processing \
  --builder kevinjieuw114514/python-builder-slim  --version 3 --poolsize 1

# poolmgr type
fission env delete --name ml-image-processing
fission env delete --name yolo
fission fn delete --name ml-image-processing
fission fn delete --name ml-object-detection
fission environment create --name ml-image-processing --image kevinjieuw114514/python-ml-image-processing \
  --builder kevinjieuw114514/python-builder-slim  --version 3 --poolsize 1 --minmemory 256 --maxmemory 512 --mincpu 500 --maxcpu 800
fission fn create --name ml-image-processing --pkg ml-image-processing --entrypoint "ml_image_processing.main" --env ml-image-processing --con 50 --requestsperpod 5
fission fn update --name ml-image-processing --con 500 --requestsperpod 10
fission environment update --name ml-image-processing --minmemory 100 --maxmemory 100 --mincpu 100 --maxcpu 100
fission environment update --name ml-image-processing --minmemory 256 --maxmemory 512 --mincpu 2000 --maxcpu 8000
#####################################################################
fission environment create --name yolo --image kevinjieuw114514/python-torch-yolo \
  --builder kevinjieuw114514/python-builder-slim  --version 3 --poolsize 6 --minmemory 256 --maxmemory 512 --mincpu 8000 --maxcpu 16000
fission environment update --name yolo --poolsize 6 --mincpu 2000 --maxcpu 8000 --minmemory 256 --maxmemory 512
fission fn create --name ml-object-detection --pkg ml-object-detection-yolo --entrypoint "ml_object_detection.main" --env yolo --con 500 --requestsperpod 5
fission fn update --name ml-object-detection --con 500 --requestsperpod 12
#####################################################################
fission environment update --name ml-image-processing --poolsize 20
fission environment update --name yolo --poolsize 10



# debug
kubectl get events --sort-by=.lastTimestamp -A | egrep -i 'evict|preempt|pressure|oom|killed|sandbox|cni|failed'
kubectl describe node intel-worker1 | egrep -i 'Pressure|Evict|DiskPressure|MemoryPressure|PIDPressure'
kubectl get pod -l environmentName=yolo -o wide



# ml-image-processing
zip -jr image-processing.zip ./image_processing
fission pkg update --sourcearchive image-processing.zip \
  --env ml-image-processing --buildcmd "./build.sh" --name ml-image-processing
fission pkg delete --name ml-image-processing
fission pkg create --sourcearchive image-processing.zip \
  --env ml-image-processing --buildcmd "./build.sh" --name ml-image-processing
fission pkg info --name ml-image-processing > log.log
fission fn delete --name ml-image-processing
sleep 2
fission fn create --name ml-image-processing --pkg ml-image-processing --entrypoint "ml_image_processing.main" --env ml-image-processing \
  --executortype newdeploy --minscale 1 --maxscale 50 --mincpu 500 \
  --maxcpu 800 --minmemory 256 --maxmemory 512 --targetcpu 70
fission fn update --name ml-image-processing  \
  --minscale 5 --maxscale 30
fission route create --name ml-image-processing \
  --function ml-image-processing --url /ml-image-processing --method POST

fission fn update --name ml-image-processing  \
  --mincpu 300 --maxcpu 300 --minmemory 60 --maxmemory 60

# # ml-object-detection
# zip -jr object-detection.zip ./object_detection
# fission pkg update --sourcearchive object-detection.zip \
#   --env pytorch --buildcmd "./build.sh" --name ml-object-detection
# fission pkg delete --name ml-object-detection
# fission pkg create --sourcearchive object-detection.zip \
#   --env pytorch --buildcmd "./build.sh" --name ml-object-detection
# fission pkg info --name ml-object-detection > log.log
# fission fn delete --name ml-object-detection
# fission fn create --name ml-object-detection --pkg ml-object-detection --entrypoint "ml_object_detection.main" --env pytorch \
#   --executortype newdeploy \--minscale 5 --maxscale 150 --mincpu 2000 \
#   --maxcpu 3500 --minmemory 256 --maxmemory 512 --targetcpu 50
# fission fn update --name ml-object-detection \
#   --minscale 5 --maxscale 30 
# fission route create --name ml-object-detection \
#   --function ml-object-detection --url /ml-object-detection --method POST


# ml-object-detection-yolo
zip -jr object-detection-yolo.zip ./object_detection
fission pkg update --sourcearchive object-detection-yolo.zip \
  --env yolo --buildcmd "./build.sh" --name ml-object-detection-yolo
fission pkg delete --name ml-object-detection-yolo
fission pkg create --sourcearchive object-detection-yolo.zip \
  --env yolo --buildcmd "./build.sh" --name ml-object-detection-yolo
fission pkg info --name ml-object-detection-yolo > log.log
fission fn delete --name ml-object-detection
sleep 2.5
fission fn create --name ml-object-detection --pkg ml-object-detection-yolo --entrypoint "ml_object_detection.main" --env yolo \
  --executortype newdeploy --minscale 1 --maxscale 100 --mincpu 2000 \
  --maxcpu 6000 --minmemory 256 --maxmemory 512 --targetcpu 70
fission fn update --name ml-object-detection \
  --minscale 5 --maxscale 30 
fission route delete --name ml-object-detection
fission route create --name ml-object-detection \
  --function ml-object-detection --url /ml-object-detection --method POST

# fast-creation
fission fn delete --name ml-image-processing
sleep 2.5
fission fn create --name ml-image-processing --pkg ml-image-processing --entrypoint "ml_image_processing.main" --env ml-image-processing \
  --executortype newdeploy --minscale 1 --maxscale 50 --mincpu 500 \
  --maxcpu 800 --minmemory 50 --maxmemory 70 --targetcpu 70

fission fn delete --name ml-object-detection
sleep 2.5
fission fn create --name ml-object-detection --pkg ml-object-detection-yolo --entrypoint "ml_object_detection.main" --env yolo \
  --executortype newdeploy --minscale 1 --maxscale 100 --mincpu 2000 \
  --maxcpu 6000 --minmemory 350 --maxmemory 512 --targetcpu 70

# keda
helm repo add kedacore https://kedacore.github.io/charts
kubectl create namespace keda
helm install keda kedacore/keda --namespace keda

# topic trigger, (install keda first, search "keda" below)
fission mqtrigger delete --name ml-image-processing
sleep 2
fission mqtrigger create \
  --name ml-image-processing \
  --function ml-object-detection \
  --mqtype redis \
  --mqtkind keda \
  --topic ml-image-processing \
  --errortopic ml-image-processing-error-topic \
  --maxretries 3 \
  --maxreplicacount 20 \
  --minreplicacount 20 \
  --metadata address=redis.ot-operators.svc.cluster.local:6379 \
  --metadata listName=ml-image-processing \
  --metadata listLength="10"


# aquatope query/test workflow creation
cd ~/uncertain-carbon/functions
# create 10 functions (1 to 10) for parallelism testing
fission fn create --name ml-image-processing-test1 --pkg ml-image-processing --entrypoint "ml_image_processing.main" --env ml-image-processing \
  --executortype newdeploy --minscale 1 --maxscale 1 --mincpu 2000 \
  --maxcpu 2000 --minmemory 256 --maxmemory 512 --targetcpu 70
fission fn create --name ml-object-detection-test1 --pkg ml-object-detection-yolo --entrypoint "ml_object_detection.main" --env yolo \
  --executortype newdeploy --minscale 1 --maxscale 1 --mincpu 2000 \
  --maxcpu 2000 --minmemory 256 --maxmemory 512 --targetcpu 70


# error topic inspection
# 1. check
kubectl -n ot-operators run -it --rm redistest \
  --image=redis:7-alpine --restart=Never -- \
  redis-cli -h redis.ot-operators.svc.cluster.local -p 6379 LRANGE ml-image-processing-error-topic 0 -1
# 2. inspect
kubectl -n ot-operators run -it --rm redistest \
  --image=redis:7-alpine --restart=Never -- \
  redis-cli -h redis.ot-operators.svc.cluster.local -p 6379 LRANGE ml-image-processing-error-topic 0 -1
# 3. temp curl
kubectl run -it --rm tmpcurl --image=curlimages/curl --restart=Never -n default -- \
  curl -sS -i -X POST http://router.fission/fission-function/ml-object-detection \
  -H 'Content-Type: application/json' \
  -d '{"image_name":"000b7b74-0a22-4d0c-b717-e240fdc5d555.png","req_id":"6d0e7cc1-f981-4aad-9484-c6acfc6360cc","duration":{"ml-image-processing":0.14236813806928694}}'


# ml-pipeline numa setting
POD=$(kubectl -n default get pod -l app=newdeploy-ml-object-detection-default -o jsonpath='{.items[0].metadata.name}')
kubectl -n default exec "$POD" -- printenv | egrep 'OMP_|GOMP_'

kubectl get deployments.apps newdeploy-ml-object-detection-default-b469-01a8ab59b0c9 -o yaml > ml-object-detection.yaml
kubectl get deployments.apps newdeploy-ml-object-detection-default-b469-01a8ab59b0c9 -o yaml > hu.yaml
############################
spec:
      containers:
      - env:
        - name: RESOURCE_VERSION_COUNT
          value: "0"
        - name: OMP_NUM_THREADS
          value: "8"
        - name: OMP_PROC_BIND
          value: "true"
        - name: OMP_PLACES
          value: "cores"
        - name: GOMP_SPINCOUNT
          value: "0"
        image: kevinjieuw114514/python-torch-yolospec:
      containers:
      - env:
        - name: RESOURCE_VERSION_COUNT
          value: "0"
        - name: OMP_NUM_THREADS
          value: "8"
        - name: OMP_PROC_BIND
          value: "true"
        - name: OMP_PLACES
          value: "cores"
        - name: GOMP_SPINCOUNT
          value: "0"
        image: kevinjieuw114514/python-torch-yolo
env:
  - name: OMP_NUM_THREADS
    value: "2"
  - name: OMP_PROC_BIND
    value: "true"
  - name: OMP_PLACES
    value: "cores"
  - name: OMP_WAIT_POLICY
    value: "PASSIVE"
  - name: MKL_NUM_THREADS
    value: "2"
  - name: OPENBLAS_NUM_THREADS
    value: "2"
  - name: NUMEXPR_MAX_THREADS
    value: "2"
  - name: OPENCV_NUM_THREADS
    value: "0"   # OpenCV: 0 often means "disable internal threading" (depending on build)
  - name: KMP_BLOCKTIME
    value: "0"

########################
kubectl apply -f ml-object-detection.yaml
kubectl rollout restart deployment newdeploy-ml-object-detection-default-b469-01a8ab59b0c9
kubectl rollout status deployment newdeploy-ml-object-detection-default-b469-01a8ab59b0c9

kubectl -n default patch deployment newdeploy-ml-object-detection-default \
  --patch-file ml-object-detection.yaml

sudo mkdir -p /etc/rancher/k3s
sudo nano /etc/rancher/k3s/config.yaml
kubelet-arg:
  - "cpu-manager-policy=static"
  - "cpu-manager-policy-options=full-pcpus-only=true"
  - "topology-manager-policy=restricted"

sudo systemctl restart k3s-agent
sudo systemctl status k3s-agent --no-pager
sudo journalctl -u k3s-agent -b --no-pager -n 200


kubectl drain intel-worker1 --ignore-daemonsets --delete-emptydir-data --force


# patch for HPA behavior
kubectl patch hpa newdeploy-ml-image-processing-default-9b09-b29e7fe83013 \
  --type=merge \
  --patch-file hpa-behavior-patch.yaml
kubectl get horizontalpodautoscalers -o yaml

sudo mkdir -p /etc/rancher/k3s
sudo vim /etc/rancher/k3s/config.yaml
## kube-controller-manager-arg:
##  - "horizontal-pod-autoscaler-sync-period=10s"
sudo systemctl restart k3s
ps aux | grep kube-controller-manager | grep horizontal-pod-autoscaler-sync-period

kubectl -n kube-system edit deployment metrics-server
# - --metric-resolution=15s
kubectl -n kube-system rollout restart deployment metrics-server
kubectl -n kube-system rollout status deployment metrics-serve


fission mqtrigger delete --name ml-image-processing
fission mqtrigger create \
  --name ml-pipeline-result \
  --mqtype redis \
  --mqtkind keda \
  --topic ml-pipeline-result \
  --errortopic ml-pipeline-result-error-topic \
  --maxretries 3 \
  --metadata address=redis.ot-operators.svc.cluster.local:6379 \
  --metadata listName=ml-pipeline-result \
  --metadata listLength="30"

# check affinity
kubectl exec -n default -it <pod> -- sh -c '
echo "---- affinity-related env ----";
env | egrep -i "^(OMP|KMP|GOMP|MKL|OPENBLAS|NUMEXPR|DNNL|ONEDNN|OMP_)" | sort || true
'
kubectl exec -n default -it newdeploy-ml-object-detection-default-884d-59d05a86d3a4-58ldksg -- sh -c ' echo "PID 1 affinity:"; taskset -pc 1 2>/dev/null || true echo; echo "Top threads by CPU:"; ps -L -p 1 -o pid,tid,psr,pcpu,comm --sort=-pcpu | head -20 echo; echo "Affinity per thread (first 20 tids):"; for t in /proc/1/task/*; do tid=$(basename "$t"); echo -n "$tid "; grep Cpus_allowed_list "$t/status" | awk "{print \$2}"; done | head -20 '

# OpenTelemety
# cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/latest/download/cert-manager.yaml
# open telemetry operator
kubectl apply -f https://github.com/open-telemetry/opentelemetry-operator/releases/latest/download/opentelemetry-operator.yaml
# OpenTelemetry deployment
kubectl apply -f opentelemetry.yaml
# Jaeger operator
kubectl create namespace observability
kubectl create -n observability -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.39.0/jaeger-operator.yaml
# Jaeger instance
kubectl apply -n observability -f jaeger.yaml
helm upgrade fission fission-charts/fission-all --namespace fission -f fission-opentelemetry.yaml
python -m pip install opentelemetry-sdk

# argo
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.7.6/install.yaml

# redis
kubectl create namespace ot-operators
helm repo add ot-helm https://ot-container-kit.github.io/helm-charts/
helm upgrade redis-operator ot-helm/redis-operator \
    --install --namespace ot-operators
# helm upgrade redis ot-helm/redis --install --namespace ot-operators
helm -n ot-operators upgrade redis ot-helm/redis   --reuse-values   --set externalService.enabled=true   --set \
  externalService.serviceType=NodePort   --set externalService.port=6379
kubectl -n ot-operators patch svc redis-external-service -p '{"spec":{"type":"NodePort", "ports":[{"port":6379,"targetPort":6379,"nodePort":32204}]}}'

# minio
helm repo add minio https://charts.min.io/
helm repo update

kubectl create namespace minio

kubectl label node intel-manager minio-node=true --overwrite

helm install minio minio/minio \
  --namespace minio \
  -f minio-values.yaml

wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
./mc --help
sudo mv mc /usr/local/bin/mc

export POD_NAME=$(kubectl get pods --namespace minio -l "release=minio" -o jsonpath="{.items[0].metadata.name}")
kubectl port-forward $POD_NAME 9000 --namespace minio
export MC_HOST_minio_local=http://$(kubectl get secret --namespace minio minio \
  -o jsonpath="{.data.rootUser}" | base64 --decode):$(kubectl get secret \
  --namespace minio minio -o jsonpath="{.data.rootPassword}" | \
  base64 --decode)@localhost:9000

kubectl port-forward -n minio svc/minio-console 9001
# service NodePort patching
kubectl patch svc minio -n minio \
  -p '{"spec": {"type": "NodePort", "ports": [{"port": 9000, "targetPort": 9000, "nodePort": 30900}]}}'

mc alias set myminio http://localhost:30900 minioadmin minioadmin123
mc mb myminio/images
mc ls myminio
mc ls myminio/images
mc ls myminio/processed-images
mc mb myminio/processed-images
mc rb myminio/processed-images
## image
wget https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-000001.zip?download=true
mkdir images
unzip part-000001.zip?download=true -d images
rm images/part-000001.json

mc cp images/* myminio/images



kubectl -n monitoring patch svc prometheus-kube-prometheus-prometheus \
  --type='merge' \
  -p '{
    "spec": {
      "type": "NodePort",
      "ports": [
        { "name": "http-web", "port": 9090, "targetPort": 9090, "nodePort": 30990, "protocol": "TCP" },
        { "name": "http-reloader", "port": 8080, "targetPort": 8080, "protocol": "TCP" }
      ]
    }
  }'

kubectl patch svc minio-console -n minio \
  -p '{"spec": {"type": "NodePort", "ports": [{"port": 9001, "targetPort": 9001, "nodePort": 30901}]}}'



# patch k8s services to NodePort
kubectl -n monitoring patch svc prometheus-grafana -p '{"spec":{"type":"NodePort"}}'
kubectl -n observability patch svc jaeger-query -p '{"spec":{"type":"NodePort"}}'

kubectl patch svc minio -n minio \
  -p '{"spec": {"type": "NodePort", "ports": [{"port": 9000, "targetPort": 9000, "nodePort": 30900}]}}'


# grafana password
kubectl get secret --namespace monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
# 8DsiRxZUQkQgTGlVDU0UwbYAbzNNDc31UYy9694U



# fission cli
curl -Lo fission https://github.com/fission/fission/releases/download/v1.22.0/fission-v1.22.0-linux-amd64 \
    && chmod +x fission && sudo mv fission /usr/local/bin/
fission version

# prometheus
# 1) Add Prometheus community charts and update
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# 2) Create a namespace for metrics stack
kubectl create namespace monitoring

# 3) Install kube-prometheus-stack into `monitoring` namespace
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring

kubectl -n monitoring get svc | grep prometheus

base64 <<EOF
canary:
  enabled: true
  prometheusSvc: "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"
EOF

kubectl -n fission patch configmap feature-config \
    -p '{"data":{"config.yaml":"Y2FuYXJ5OgogIGVuYWJsZWQ6IHRydWUKICBwcm9tZXRoZXVzU3ZjOiAiaHR0cDovL3Byb21ldGhldXMta3ViZS1wcm9tZXRoZXVzLXByb21ldGhldXMubW9uaXRvcmluZy5zdmMuY2x1c3Rlci5sb2NhbDo5MDkwIgo="}}'

kubectl -n fission get deploy -o name | xargs -n1 kubectl -n fission rollout restart


# kn-cli
curl -SLf https://github.com/knative/client/releases/download/knative-v1.20.0/kn-linux-amd64 > kn
sudo chmod 777 kn
sudo mv kn /usr/local/bin/kn
kn version

# kn-operator
curl -SLf https://github.com/knative-extensions/kn-plugin-operator/releases/download/knative-v1.7.1/kn-operator-linux-amd64 > kn-operator
sudo chmod 777 kn-operator
mkdir -p ~/.config/kn/plugins
cp kn-operator ~/.config/kn/plugins
kn operator -h


# We recommend creating two namespaces, one for the 'OpenFaaS core services' and one for the 'functions'.
kubectl apply -f https://raw.githubusercontent.com/openfaas/faas-netes/master/namespaces.yml

helm repo add openfaas https://openfaas.github.io/faas-netes/

helm repo update \
 && helm upgrade openfaas \
  --install openfaas/openfaas \
  --namespace openfaas

PASSWORD=$(kubectl -n openfaas get secret basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode) && \
echo "OpenFaaS admin password: $PASSWORD"

kubectl get svc -n openfaas gateway-external -o wide
CLUSTER_IP=10.52.2.108
echo $CLUSTER_IP
OPENFAAS_URL=$CLUSTER_IP:31112
export OPENFAAS_URL=$OPENFAAS_URL
echo "export OPENFAAS_URL=$OPENFAAS_URL" >> ~/.bashrc
source ~/.bashrc

echo -n $PASSWORD | faas-cli login -g $OPENFAAS_URL -u admin -p $PASSWORD

# Download the latest release
VERSION=$(curl -s https://api.github.com/repos/containerd/nerdctl/releases/latest | grep tag_name | cut -d '"' -f 4)
curl -LO https://github.com/containerd/nerdctl/releases/download/${VERSION}/nerdctl-${VERSION#v}-linux-amd64.tar.gz

# Extract
sudo tar zxvf nerdctl-*-linux-amd64.tar.gz -C /usr/local/bin nerdctl
nerdctl login docker.io

kubectl patch svc prometheus -n openfaas \
  -p '{"spec": {"type": "NodePort", "ports": [{"port": 9090, "targetPort": 9090, "nodePort": 30091}]}}'

curl -u admin:$PASSWORD \
  http://10.52.2.108:31112/system/functions

kubectl patch svc alertmanager -n openfaas \
  -p '{"spec": {"type": "NodePort", "ports": [{"port": 9093, "targetPort": 9093, "nodePort": 30093}]}}'

# conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n faas python=3.10
conda activate faas
python -m pip install locust

# autoscaler rule
kubectl -n openfaas get configmap prometheus-config -o yaml > prometheus-config.yaml
kubectl -n openfaas apply -f prometheus-config.yaml
kubectl -n openfaas delete pod -l app=prometheus
kubectl -n openfaas get pods

kubectl -n openfaas-fn get deploy my-function -o yaml > hu.yaml

kubectl -n openfaas get configmap alertmanager-config -o yaml > alertmanager-config.yaml
kubectl -n openfaas apply -f alertmanager-config.yaml
kubectl -n openfaas delete pod -l app=alertmanager
kubectl -n openfaas get pods




sudo apt-get update
sudo apt-get install -y containerd
sudo apt-get install -y net-tools
# Create default config and switch to systemd cgroups
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml >/dev/null
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl restart containerd
sudo systemctl enable containerd


# Add Kubernetes apt repo
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl gpg
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.30/deb/Release.key \
  | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.30/deb/ /' \
  | sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# bash completion (helpful!)
sudo apt-get install -y bash-completion
echo 'source <(kubectl completion bash)' >>~/.bashrc
source ~/.bashrc



# docker
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update

sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo chmod 777 /var/run/docker.sock
# end of docker

# manager
# Kubernetes control-plane
sudo firewall-cmd --add-port=6443/tcp --permanent         # API server
sudo firewall-cmd --add-port=2379-2380/tcp --permanent    # etcd
sudo firewall-cmd --add-port=10250/tcp --permanent        # kubelet
sudo firewall-cmd --add-port=10257/tcp --permanent        # kube-controller-manager
sudo firewall-cmd --add-port=10259/tcp --permanent        # kube-scheduler

# Optional: NodePort range for services you expose that way
sudo firewall-cmd --add-port=30000-32767/tcp --permanent

# CNI-dependent (pick what matches your plugin)
# Calico (VXLAN default): 
sudo firewall-cmd --add-port=4789/udp --permanent         # VXLAN
# If using Calico BGP mode instead of VXLAN:
sudo firewall-cmd --add-port=179/tcp --permanent          # BGP

# Flannel (VXLAN):
# sudo firewall-cmd --add-port=8472/udp --permanent

sudo firewall-cmd --reload


sudo firewall-cmd --list-all-zones


# worker:
sudo firewall-cmd --add-port=10250/tcp --permanent
sudo firewall-cmd --add-port=30000-32767/tcp --permanent   # NodePort (optional)

# Match your CNI choice:
# Calico VXLAN:
sudo firewall-cmd --add-port=4789/udp --permanent
# If using Calico BGP mode instead of VXLAN:
sudo firewall-cmd --add-port=179/tcp --permanent          # BGP
# Flannel VXLAN:
# sudo firewall-cmd --add-port=8472/udp --permanent

sudo firewall-cmd --reload

sudo firewall-cmd --list-all-zones

# On k8s-mgr
sudo kubeadm reset -f
sudo kubeadm init --pod-network-cidr=192.168.0.0/16
mkdir -p $HOME/.kube
sudo cp /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
export KUBECONFIG=/etc/kubernetes/admin.conf

kubeadm join 10.191.131.3:6443 --token ujsmyk.bjuzr96aszywjg0c \
    --discovery-token-ca-cert-hash sha256:02c1f427bf6986dbd7a40de8e5cbcfb4a5977f9b183d1d2bb39b5e2193880075 

# test
kubectl get nodes

# CNI, manager only
# Download Calico manifest
# curl -fsSL -O https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/calico.yaml

# replace the default 192.168.0.0/16 pool with 10.52.0.0/16
# sed -i 's#192\.168\.0\.0/16#10.52.0.0/16#g' calico.yaml
kubectl apply -f calico.yaml

# create token on manager
kubeadm token create --print-join-command

# join workers
sudo kubeadm join <CONTROL_PLANE_IP>:6443 \
  --token <token> \
  --discovery-token-ca-cert-hash sha256:<hash>

# mgr smoke test
kubectl get nodes
kubectl get pods -A

wget https://get.helm.sh/helm-v3.19.0-linux-amd64.tar.gz
tar -xzvf helm-v3.19.0-linux-amd64.tar.gz
sudo mv linux-amd64/helm /usr/local/bin/helm
helm version
rm -rf helm-v3.19.0-linux-amd64.tar.tar.gz linux-amd64

helm repo add openwhisk https://openwhisk.apache.org/charts
helm repo update

sudo mkdir -p /opt/local-path-provisioner
sudo chmod 777 /opt/local-path-provisioner   # quick unblock; tighten later

kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/master/deploy/local-path-storage.yaml

kubectl -n local-path-storage rollout restart deploy/local-path-provisioner
kubectl -n local-path-storage rollout status deploy/local-path-provisioner

kubectl -n local-path-storage get pods

kubectl apply -f pvc-test.yaml
kubectl get pvc pvc-test       # should show STATUS=Bound
kubectl exec -it pvc-tester -- cat /data/hello
kubectl delete pod pvc-tester
kubectl get pods

kubectl label node intel-worker1 openwhisk-role=invoker
kubectl label node intel-manager openwhisk-role=core

kubectl annotate storageclass local-path storageclass.kubernetes.io/is-default-class="true" --overwrite

kubectl create namespace openwhisk
helm install ow openwhisk/openwhisk -n openwhisk -f values.yaml
helm uninstall ow -n openwhisk
kubectl delete namespace openwhisk
kubectl get pods -n openwhisk

# uninstall
sudo /usr/local/bin/k3s-uninstall.sh || true
sudo /usr/local/bin/k3s-agent-uninstall.sh || true

sudo rm -rf \
  /var/lib/kubelet \
  /var/lib/rancher \
  /etc/rancher \
  /run/k3s \
  /run/flannel


# 神秘小指令
kubectl patch hpa newdeploy-ml-image-processing-default-b931-31483219
d708 \
  --type=merge \
  -p '{
    "spec": {
      "minReplicas": 1,
      "maxReplicas": 1,
      "metrics": [
        {
          "type": "Resource",
          "resource": {
            "name": "cpu",
            "target": {
              "type": "Utilization",
              "averageUtilization": 70
            }
          }
        }
      ]
    }
  }'

kubectl patch hpa newdeploy-ml-object-detection-default-b995-7cfe76e42031 \
  --type=merge \
  -p '{
    "spec": {
      "minReplicas": 1,
      "maxReplicas": 100,
      "metrics": [
        {
          "type": "Resource",
          "resource": {
            "name": "cpu",
            "target": {
              "type": "Utilization",
              "averageUtilization": 50
            }
          }
        }
      ]
    }
  }'


kubectl scale deployment newdeploy-ml-object-detection-default-aac7-4da75d82037c --replicas=60
kubectl scale deployment newdeploy-ml-image-processing-default-9a0e-f899636d0d4a --replicas=6

# metrics
kubectl exec -it redis-0 -n ot-operators -- redis-cli --latency
kubectl top pod -n ot-operators -l app=redis

# throttle check
kubectl -n default get pod newdeploy-ml-object-detection-default-a709-07b914aa9baa-68xt9gn   -o custom-columns=NAME:.metadata.name,CPU_REQ:.spec.containers[*].resources.requests.cpu,CPU_LIM:.spec.containers[*].resources.limits.cpu
kubectl exec -n default -it newdeploy-ml-object-detection-default-a709-07b914aa9baa-68zvbnj -- sh -c '
echo "cpu.max:"; cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
echo "cpu.stat:"; cat /sys/fs/cgroup/cpu.stat 2>/dev/null || true
'