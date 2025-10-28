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

# On k8s-mgr
sudo kubeadm init --pod-network-cidr=10.52.0.0/16

mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# CNI, manager only
curl -fsSL -O https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/calico.yaml
# replace the default 192.168.0.0/16 pool with 10.52.0.0/16
sed -i 's#192\.168\.0\.0/16#10.52.0.0/16#g' calico.yaml
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
