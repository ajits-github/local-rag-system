# Kubernetes Operational Exercises (minikube)

Run these against the local `rag-learning` minikube cluster from
`README.md` in this directory (`kubectl config use-context rag-learning`
first if you've been using another cluster context).

`kubectl` mechanics don't know or care which local cluster they're
talking to, so 17 of the original 18 exercises in `../kind/EXERCISES.md`
work here completely unchanged: same commands, same expected behavior,
same interview answers. Use that file directly. This file only records
where minikube's single-node `docker` driver genuinely changes the
exercise, so nothing here duplicates content that's identical either
way.

---

## Exercise 12 ("Kill a worker node"): not reproducible here

This is the one exercise that cannot be demonstrated on this
environment. It needs a genuinely separate second node to stop while the
control plane keeps running. This cluster only has one node, acting as
both control-plane and worker.

Do not run `docker stop rag-learning` as a substitute. Unlike kind's
`docker stop rag-learning-worker2` (which stops one of three nodes,
leaving the control plane running elsewhere to observe), stopping this
cluster's only node stops everything, including the API server you'd be
using to observe the "failure." That's a full outage, not a node-failure
simulation, and it teaches a different (much less interesting) lesson.

Run `../kind/EXERCISES.md` exercise 12 on the kind cluster instead if
you want to see this one hands-on. If `kind` doesn't work on your host
(see `../kind/README.md`'s note and `ISSUES.md`), this exercise stays a
read-only/conceptual exercise here: reason through the "unreachable node
vs. confirmed-dead node" distinction from the write-up in
`../kind/EXERCISES.md` without running it, and be ready to explain it in
an interview even without a live reproduction.

---

## Exercise 3 ("Scale rag-api from 1 to 3 replicas"): one caveat

The "important correction" about the scheduler not necessarily spreading
replicas evenly across nodes is trivially true here for a different
reason than on kind: there's only one node, so every replica lands on
it by construction. There is no spreading decision to observe at all.
The Service/ClusterIP/EndpointSlice behavior the exercise is actually
about is unaffected and works exactly as described.

---

## Everything else

Exercises 1, 2, 4-11, 13-19 in `../kind/EXERCISES.md`: run as written,
substituting nothing except the cluster you're pointed at. The debugging
drills, the inspection toolkit, and the interview recap table all apply
unchanged.
