import torch

from model import Delphi


class DelphiMod(Delphi):
    @torch.no_grad()
    def generate(
        self,
        idx,
        age,
        max_new_tokens=100,
        max_age=85 * 365.25,
        no_repeat=True,
        termination_tokens=None,
        top_k=None,
    ):
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64)
        mask_time = -10000

        if max_new_tokens == -1:
            max_new_tokens = 128

        t_list = []

        for _ in range(max_new_tokens):
            logits, _, _ = self(idx, age)
            logits = logits[:, -1, :]
            logits[:, self.config.ignore_tokens] = -torch.inf

            if no_repeat:
                fill = idx.clone()
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)

            # sample from exponential distributions for each disease using the inverse CDF method, then take min
            t = torch.clamp(
                -torch.exp(-logits) * torch.rand(logits.shape).log(),
                min=0,
                max=365 * 80,
            )
            t_next = t.min(1)
            idx_next = t_next[1][:, None]  # the index of the min sampled time
            age_next = (
                age[..., [-1]] + t_next[0][:, None]
            )  # the value of the min sampled time

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)

            t_list.append(t)

            if torch.logical_or(
                torch.isin(idx, termination_tokens).any(-1), age_next > max_age
            ).all():
                break

        pad = (
            torch.cumsum(
                torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(),
                1,
            )
            > 1
        ) + (age > max_age)

        logits, _, _ = self(idx, age)
        idx[pad] = 0
        age[pad] = mask_time

        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack(
                [
                    logits[:, j].scatter_(1, fill[:, : j + 1], -torch.inf)
                    for j in range(fill.shape[1])
                ]
            ).transpose(0, 1)

        return idx.cpu(), age.cpu(), logits.cpu(), torch.stack(t_list, dim=1).cpu()
