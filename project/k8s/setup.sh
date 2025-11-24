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


sudo apt-get update
sudo apt-get install -y containerd
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
sudo kubeadm init --pod-network-cidr=192.168.0.0/16
mkdir -p $HOME/.kube
sudo cp /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
export KUBECONFIG=/etc/kubernetes/admin.conf

kubeadm join 10.191.131.3:6443 --token 4a8gpc.gsdmu0cikrvd3pn2 \
        --discovery-token-ca-cert-hash sha256:5f75de8a8024141916af1ff105b511d84e8851fce17b29beefeec542e0da7c7f 

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
rm -rf helm-v4.0.0-beta.2-linux-amd64.tar.gz linux-amd64

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
