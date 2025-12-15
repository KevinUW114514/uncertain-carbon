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

curl -sfL https://get.k3s.io | sudo sh -
sudo cat /var/lib/rancher/k3s/server/node-token

sudo chmod 777 /etc/rancher/k3s/k3s.yaml

mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config

sudo apt update
sudo apt install -y bash-completion net-tools

kubectl get pods -A

# workers:
curl -sfL https://get.k3s.io | \
  sudo sh -s - agent \
    --server https://10.52.2.108:6443 \
    --token "K103b0b11b947418327eba9e397c68e27255a864a55b07fffd7ff0edadf4eed1376::server:afbe48d958e93bb8edc34651414a0d24"

sudo chmod 777 /etc/rancher/k3s/k3s.yaml
kubectl get nodes

# helm
curl -sSLf https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash

# fission
export FISSION_NAMESPACE="fission"
kubectl create namespace $FISSION_NAMESPACE
kubectl create -k "github.com/fission/fission/crds/v1?ref=v1.21.0"
helm repo add fission-charts https://fission.github.io/fission-charts/
helm repo update
helm install --version v1.21.0 --namespace $FISSION_NAMESPACE fission \
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

docker build -f Dockerfile-debian -t kevinjieuw114514/python-builder-slim .
docker push kevinjieuw114514/python-builder-slim

# v3 custom env
fission environment create --name pytorch --image kevinjieuw114514/python-env-slim \
  --builder kevinjieuw114514/python-builder-slim  --version 3 --poolsize 1

# ml-image-processing
zip -jr image-processing.zip ./image_processing
fission pkg update --sourcearchive image-processing.zip \
  --env python --buildcmd "./build.sh" --name ml-image-processing
fission pkg info --name ml-image-processing > log.log
fission fn delete --name ml-image-processing
fission fn create --name ml-image-processing --pkg ml-image-processing --entrypoint "ml_image_processing.main" --env python \
  --executortype newdeploy \--minscale 5 --maxscale 15 --mincpu 1000 \
  --maxcpu 1500 --minmemory 256 --maxmemory 512 --targetcpu 50
fission route create --name ml-image-processing \
  --function ml-image-processing --url /ml-image-processing --method POST

# ml-object-detection
zip -jr object-detection.zip ./object_detection
fission pkg delete --name ml-object-detection
fission pkg create --sourcearchive object-detection.zip \
  --env pytorch --buildcmd "./build.sh" --name ml-object-detection
fission pkg info --name ml-object-detection > log.log
fission fn delete --name ml-object-detection
fission fn create --name ml-object-detection --pkg ml-object-detection --entrypoint "ml_object_detection.main" --env pytorch \
  --executortype newdeploy \--minscale 5 --maxscale 15 --mincpu 2000 \
  --maxcpu 3500 --minmemory 256 --maxmemory 512 --targetcpu 50
fission route create --name ml-object-detection \
  --function ml-object-detection --url /ml-object-detection --method POST

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



# minio
helm repo add minio https://charts.min.io/
helm repo update

kubectl create namespace minio

helm install minio minio/minio \
  --namespace minio \
  -f minio-values.yaml

wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
./mc --help
sudo mv mc /usr/local/bin/mc

export POD_NAME=$(kubectl get pods --namespace minio -l "release=minio" -o jsonpath="{.items[0].metadata.name}")
kubectl port-forward $POD_NAME 9000 --namespace minio
export MC_HOST_minio-local=http://$(kubectl get secret --namespace minio minio \
  -o jsonpath="{.data.rootUser}" | base64 --decode):$(kubectl get secret \
  --namespace minio minio -o jsonpath="{.data.rootPassword}" | \
  base64 --decode)@localhost:9000

mc alias set myminio http://localhost:30900 minioadmin minioadmin123
mc mb myminio/images
mc ls myminio
mc ls myminio/images
mc rb myminio/processed-mages

kubectl port-forward -n minio svc/minio-console 9001



# service NodePort patching
kubectl patch svc minio -n minio \
  -p '{"spec": {"type": "NodePort", "ports": [{"port": 9000, "targetPort": 9000, "nodePort": 30900}]}}'

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

kubectl -n monitoring patch svc prometheus-grafana -p '{"spec":{"type":"NodePort"}}'

kubectl patch svc minio -n minio \
  -p '{"spec": {"type": "NodePort", "ports": [{"port": 9000, "targetPort": 9000, "nodePort": 30900}]}}'


# grafana password
kubectl get secret --namespace monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
# 8DsiRxZUQkQgTGlVDU0UwbYAbzNNDc31UYy9694U


## image
wget https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-000001.zip?download=true
mkdir images
unzip part-000001.zip?download=true -d images
rm images/part-000001.json

mc cp images/* myminio/images


# fission cli
curl -Lo fission https://github.com/fission/fission/releases/download/v1.21.0/fission-v1.21.0-linux-amd64 \
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